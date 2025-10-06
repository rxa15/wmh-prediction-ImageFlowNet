import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import numpy as np
from collections import defaultdict
import itertools
from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt
from ImageFlowNet.src.nn.imageflownet_ode import ImageFlowNetODE
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    followed by a cosine annealing schedule.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + (self.base_lrs[0] - self.warmup_start_lr) * (self.last_epoch / self.warmup_epochs)
        else:
            lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * \
                (1 + np.cos(np.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
        return [lr for _ in self.base_lrs]

def neg_cos_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity for SimSiam."""
    z = z.detach()  # Stop gradient
    p = torch.nn.functional.normalize(p, p=2, dim=1)  # L2-normalize
    z = torch.nn.functional.normalize(z, p=2, dim=1)  # L2-normalize
    return -(p * z).sum(dim=1).mean()

# --- Loss and Metric Classes ---
# NOTE: DiceLoss isn't used for training anymore, but kept for reference. MSE is used instead.
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score

class DiceMetric:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        self.intersection = 0.0
        self.union = 0.0

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        preds = torch.sigmoid(logits)
        preds = preds.view(-1)
        target = target.view(-1)
        self.intersection += (preds * target).sum()
        self.union += preds.sum() + target.sum()

    def compute(self):
        return (2. * self.intersection + self.smooth) / (self.union + self.smooth)

    def to(self, device):
        return self

    def reset(self):
        self.intersection = 0.0
        self.union = 0.0

# --- Dataset Class ---
class FLAIREvolutionDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_slices_per_patient=None):
        self.transform = transform
        print("Scanning and grouping files...")
        patient_scan_info = defaultdict(lambda: defaultdict(list))
        for root, _, filenames in os.walk(root_dir):
            if "FLAIR" not in root:
                continue
            for f in filenames:
                if f.endswith((".nii", ".nii.gz")):
                    _, patient_id, wave = self.parse_filename(f)
                    patient_scan_info[patient_id][wave].append(os.path.join(root, f))

        print("Filtering out blank slices and creating pairs...")
        self.index_map = []
        self.patient_ids = set()
        for patient_id, waves_data in tqdm(patient_scan_info.items(), desc="Processing Patients"):
            sorted_waves = sorted(waves_data.keys(), key=int)
            if len(sorted_waves) < 2:
                continue
            
            volumes = {}
            try:
                for wave in sorted_waves:
                    volumes[wave] = nib.load(waves_data[wave][0]).get_fdata(dtype=np.float32)
            except Exception as e:
                print(f"Warning: Could not load volumes for patient {patient_id}. Skipping. Error: {e}")
                continue

            for source_wave, target_wave in itertools.pairwise(sorted_waves):
                source_vol, target_vol = volumes[source_wave], volumes[target_wave]
                if source_vol.shape != target_vol.shape:
                    print(f"Warning: Mismatched shapes for patient {patient_id}, waves {source_wave}-{target_wave}. Skipping pair.")
                    continue
                
                num_slices = source_vol.shape[2]
                slice_indices = list(range(14, num_slices))
                
                if max_slices_per_patient and len(slice_indices) > max_slices_per_patient:
                    step = len(slice_indices) / max_slices_per_patient
                    slice_indices = [int(i * step) for i in range(max_slices_per_patient)]

                for s_idx in slice_indices:
                    source_slice, target_slice = source_vol[:, :, s_idx], target_vol[:, :, s_idx]
                    if source_slice.std() > 1e-6 and target_slice.std() > 1e-6:
                        time_delta = float(target_wave) - float(source_wave)
                        self.index_map.append({
                            "patient_id": patient_id,
                            "source_path": waves_data[source_wave][0],
                            "target_path": waves_data[target_wave][0],
                            "slice_idx": s_idx,
                            "time_delta": time_delta
                        })
                        self.patient_ids.add(patient_id)
        
        self.patient_ids = list(self.patient_ids)
        print(f"Dataset initialized. Found {len(self.index_map)} valid pairs from {len(self.patient_ids)} patients.")

    def parse_filename(self, filepath):
        fname = os.path.basename(filepath)
        parts = fname.split("_")
        return parts[0][:5], parts[0][5:], parts[1]

    def __len__(self):
        return len(self.index_map)

    def _load_slice(self, file_path, slice_idx):
        vol = nib.load(file_path).get_fdata(dtype=np.float32)
        img_slice = vol[:, :, slice_idx]
        min_val, max_val = img_slice.min(), img_slice.max()
        if max_val - min_val > 1e-8:
            img_slice = (img_slice - min_val) / (max_val - min_val)
        else:
            img_slice = np.zeros_like(img_slice)
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0)
        return img_tensor

    def __getitem__(self, idx):
        pair_info = self.index_map[idx]
        source_img = self._load_slice(pair_info["source_path"], pair_info["slice_idx"])
        target_img = self._load_slice(pair_info["target_path"], pair_info["slice_idx"])
        time_delta = torch.tensor(pair_info["time_delta"], dtype=torch.float32)
        slice_idx = torch.tensor(pair_info["slice_idx"], dtype=torch.long)
        return {
            "source_image": source_img,
            "target_image": target_img,
            "time_delta": time_delta,
            "patient_id": pair_info["patient_id"],
            "slice_idx": slice_idx
        }

# --- Configuration ---
ROOT_DIR = "/data/LBC1936" # TODO: CHange root dir for the dataset
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
K_FOLDS = 2
MAX_SLICES = 48
MAX_PATIENTS_PER_FOLD = 5
MAX_TEST_PATIENTS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECON_PSNR_THR = 25.0  # PSNR threshold to start training the ODE component
CONTRASTIVE_COEFF = 0.1 # Weight for the contrastive loss term

print(f"Using device: {DEVICE}")

# --- Visualization and Testing ---

def visualize_results(source, ground_truth, predicted, patient_ids, slice_indices, n_samples=5, filename="test_results.png", save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(15, n_samples * 5))
    n_samples = min(n_samples, source.shape[0])

    for i in range(n_samples):
        p_id = patient_ids[i]
        s_idx = slice_indices[i].item()
        
        plt.subplot(n_samples, 3, i * 3 + 1)
        plt.imshow(source[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Patient: {p_id}\nSource (Slice: {s_idx})")
        plt.axis('off')

        plt.subplot(n_samples, 3, i * 3 + 2)
        plt.imshow(ground_truth[i, 0].cpu().numpy(), cmap='gray')
        plt.title("Ground Truth (Wave T+Δt)")
        plt.axis('off')

        plt.subplot(n_samples, 3, i * 3 + 3)
        plt.imshow(predicted[i, 0].cpu().numpy(), cmap='gray')
        plt.title("Model Prediction")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visual results saved to {save_path}")

def test_single_model(model_path, test_loader):
    print(f"\n--- Evaluating model: {model_path} ---")
    model = ImageFlowNetODE(device=DEVICE, in_channels=1, ode_location='bottleneck', contrastive=True).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dice = DiceMetric()
    mse = torchmetrics.MeanSquaredError().to(DEVICE)
    psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Testing {os.path.basename(model_path)}")):
            source_img, target_img = batch["source_image"].to(DEVICE), batch["target_image"].to(DEVICE)
            time_deltas, slice_indices, patient_ids = batch["time_delta"].to(DEVICE), batch["slice_idx"], batch["patient_id"]
            
            t = time_deltas[0:1] 
            predicted_logits = model(source_img, t)
            predicted_probs = torch.sigmoid(predicted_logits)

            dice.update(predicted_logits, target_img)
            mse.update(predicted_probs, target_img)
            psnr.update(predicted_probs, target_img)
            ssim.update(predicted_probs, target_img)

            if i == 0:
                filename = f"results_{os.path.basename(model_path).replace('.pth', '.png')}"
                print(f"\nVisualizing results for patients {patient_ids} on slices: {slice_indices.tolist()}")
                visualize_results(source_img, target_img, predicted_probs, patient_ids, slice_indices, filename=filename)

    final_dice, final_mse = dice.compute().item(), mse.compute().item()
    final_psnr, final_ssim = psnr.compute().item(), ssim.compute().item()

    print(f"Results for {model_path}: Dice={final_dice:.4f}, PSNR={final_psnr:.4f}, SSIM={final_ssim:.4f}, MSE={final_mse:.6f}")
    return {'model_path': model_path, 'Dice': final_dice, 'MSE': final_mse, 'PSNR': final_psnr, 'SSIM': final_ssim}


def train_epoch(model, loader, optimizer, ema, mse_loss, device, epoch_idx, train_time_dependent):
    model.train()
    
    total_recon_loss = 0.0
    total_pred_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{NUM_EPOCHS} [Train]")
    for i, batch in enumerate(pbar):
        source_img, target_img = batch["source_image"].to(DEVICE), batch["target_image"].to(DEVICE)
        time_deltas = batch["time_delta"].to(DEVICE)

        optimizer.zero_grad()
        
        # --- 1. Reconstruction Loss (Trains Encoder/Decoder) ---
        # Unfreeze all model parts for reconstruction training
        if hasattr(model, 'unfreeze'):
            model.unfreeze()

        # Reconstruct source and target images by passing through AE with t=0
        source_recon = model(source_img, t=torch.zeros(1).to(device))
        target_recon = model(target_img, t=torch.zeros(1).to(device))
        
        loss_recon = mse_loss(source_recon, source_img) + mse_loss(target_recon, target_img)
        
        # Add contrastive loss (SimSiam style)
        if hasattr(model, 'simsiam_project') and hasattr(model, 'simsiam_predict'):
            z1, z2 = model.simsiam_project(source_img), model.simsiam_project(target_img)
            p1, p2 = model.simsiam_predict(z1), model.simsiam_predict(z2)
            loss_contrastive = neg_cos_sim(p1, z2)/2 + neg_cos_sim(p2, z1)/2
            loss = loss_recon + CONTRASTIVE_COEFF * loss_contrastive
        else:
            loss = loss_recon
            loss_contrastive = torch.tensor(0.0)

        loss.backward()
        optimizer.step()
        ema.update()
        
        total_recon_loss += loss.item()

        # --- 2. Prediction Loss (Trains ODE component) ---
        if train_time_dependent:
            optimizer.zero_grad()
            
            # Freeze time-independent parts (encoder/decoder)
            if hasattr(model, 'freeze_time_independent'):
                model.freeze_time_independent()

            t = time_deltas[0:1] # Assumes all samples in batch have same delta_t
            predicted_target = model(source_img, t)
            loss_pred = mse_loss(predicted_target, target_img)
            
            loss_pred.backward()
            optimizer.step()
            ema.update()

            total_pred_loss += loss_pred.item()
        
        pbar.set_postfix(
            recon_loss=total_recon_loss / (i + 1), 
            pred_loss=total_pred_loss / (i + 1) if train_time_dependent else "N/A"
        )
        
def val_epoch(model, loader, device):
    model.eval()
    
    recon_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    pred_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    with torch.no_grad():
        for batch in loader:
            source_img, target_img = batch["source_image"].to(DEVICE), batch["target_image"].to(DEVICE)
            time_deltas = batch["time_delta"].to(DEVICE)
            
            # Reconstruction PSNR
            source_recon = model(source_img, t=torch.zeros(1).to(device))
            target_recon = model(target_img, t=torch.zeros(1).to(device))
            recon_psnr_metric.update(source_recon, source_img)
            recon_psnr_metric.update(target_recon, target_img)
            
            # Prediction PSNR
            t = time_deltas[0:1]
            predicted_target = model(source_img, t)
            pred_psnr_metric.update(predicted_target, target_img)
            
    avg_recon_psnr = recon_psnr_metric.compute().item()
    avg_pred_psnr = pred_psnr_metric.compute().item()
    
    return avg_recon_psnr, avg_pred_psnr


def load_folds_from_csv(fold_csv_path):
    """Load predefined patient folds from CSV."""
    df = pd.read_csv(fold_csv_path)
    folds = {}
    for _, row in df.iterrows():
        pid = str(row["patient_ID"])
        fold = int(row["fold"])
        folds.setdefault(fold, []).append(pid)
    return folds

# Main logic
def main():
    print("Initializing dataset...")
    full_dataset = FLAIREvolutionDataset(root_dir=ROOT_DIR, max_slices_per_patient=MAX_SLICES)

    #Load fold assignments from CSV
    fold_csv = "patients_5fold.csv"
    if not os.path.exists(fold_csv):
        raise FileNotFoundError(f"Fold CSV not found at {fold_csv}. Please provide it.")

    folds_dict = load_folds_from_csv(fold_csv)
    all_folds = sorted(folds_dict.keys())
    print(f"Loaded predefined folds from {fold_csv}: {len(all_folds)} folds found.")

    for fold_idx in all_folds:
        print("\n" + "="*50 + f"\n============= FOLD {fold_idx} / {len(all_folds)} =============" + "\n" + "="*50)
        
        val_pids = folds_dict[fold_idx]
        train_pids = [pid for f, pids in folds_dict.items() if f != fold_idx for pid in pids]

        if MAX_PATIENTS_PER_FOLD:
            val_pids = val_pids[:MAX_PATIENTS_PER_FOLD]
            train_pids = train_pids[:MAX_PATIENTS_PER_FOLD]

        train_pids_set, val_pids_set = set(train_pids), set(val_pids)
        train_indices = [i for i, item in enumerate(full_dataset.index_map) if item['patient_id'] in train_pids_set]
        val_indices = [i for i, item in enumerate(full_dataset.index_map) if item['patient_id'] in val_pids_set]
        
        print(f"Fold {fold_idx}: {len(train_indices)} train samples, {len(val_indices)} val samples.")

        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = ImageFlowNetODE(device=DEVICE, in_channels=1, ode_location='bottleneck', contrastive=True).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=NUM_EPOCHS // 10, max_epochs=NUM_EPOCHS)
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
        mse_loss = nn.MSELoss()

        best_val_psnr = 0.0
        recon_good_enough = False
        model_save_path = f"model_fold_{fold_idx}.pth"

        for epoch in range(NUM_EPOCHS):
            train_epoch(model, train_loader, optimizer, ema, mse_loss, DEVICE, epoch, recon_good_enough)
            
            with ema.average_parameters():
                val_recon_psnr, val_pred_psnr = val_epoch(model, val_loader, DEVICE)
            
            print(f"Epoch {epoch+1}: Val Recon PSNR: {val_recon_psnr:.4f}, Val Pred PSNR: {val_pred_psnr:.4f}")

            if not recon_good_enough and val_recon_psnr > RECON_PSNR_THR:
                recon_good_enough = True
                print(f"Reconstruction PSNR threshold ({RECON_PSNR_THR}) reached. Starting ODE training.")

            if val_pred_psnr > best_val_psnr:
                best_val_psnr = val_pred_psnr
                torch.save(model.state_dict(), model_save_path)
                print(f"Val prediction PSNR improved to {best_val_psnr:.4f}. Model saved.")

            scheduler.step()

    print("\n" + "="*50 + "\n======= Training Complete. Starting Final Evaluation. =======" + "\n" + "="*50)

    # Combine all test patients
    test_pids = folds_dict[max(all_folds)] # last fold as test
    if MAX_TEST_PATIENTS:
        test_pids = test_pids[:MAX_TEST_PATIENTS]
    test_pids_set = set(test_pids)
    test_indices = [i for i, item in enumerate(full_dataset.index_map) if item['patient_id'] in test_pids_set]
    test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test data ready: {len(test_indices)} samples from {len(test_pids_set)} patients.")

    model_paths = [f"model_fold_{i}.pth" for i in all_folds]
    all_results = [res for path in model_paths if os.path.exists(path) and (res := test_single_model(path, test_loader))]

    if not all_results:
        print("No models were evaluated.")
        return

    best_dice = max(all_results, key=lambda x: x['Dice'])
    best_psnr = max(all_results, key=lambda x: x['PSNR'])
    best_ssim = max(all_results, key=lambda x: x['SSIM'])

    print("\n" + "="*50 + "\n============= Overall Evaluation Complete =============" + "\n" + "="*50)
    for res in all_results:
        print(f"{os.path.basename(res['model_path'])} | Dice: {res['Dice']:.4f} | PSNR: {res['PSNR']:.4f} | SSIM: {res['SSIM']:.4f} | MSE: {res['MSE']:.6f}")
    print(f"\nBest (Dice): {os.path.basename(best_dice['model_path'])}")
    print(f"Best (PSNR): {os.path.basename(best_psnr['model_path'])}")
    print(f"Best (SSIM): {os.path.basename(best_ssim['model_path'])}")

if __name__ == '__main__':
    main()
