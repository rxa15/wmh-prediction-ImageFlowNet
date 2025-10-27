# -*- coding: utf-8 -*-

# === Standard Library Imports ===
import itertools
import os
import re
from collections import defaultdict

# === Third-Party Imports ===
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, Subset
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

# === MONAI Imports (Medical Imaging) ===
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

# === Local Imports ===
from ImageFlowNet.src.nn.imageflownet_ode import ImageFlowNetODE


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

# --- 3D Augmentation Transforms ---
def get_train_transforms():
    """
    Returns a composition of 3D MONAI augmentations to be applied to full volumes.
    These augmentations work on 3D data before slicing occurs.
    """
    return Compose(
        transforms=[
            RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),  # radians (~5.7 degrees)
                scale_range=(0.1, 0.1, 0.1),   # ±10% scale
                mode=("nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        ],
    )

def get_val_transforms():
    """
    Returns minimal transforms for validation (no augmentation, only compatibility).
    """
    return Compose(transforms=[])

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
    """
    Custom dataset for LBC1936 folder structure where each folder encodes a scan pair:
    e.g., Scan1Wave2_FLAIR_brain, Scan1Wave2_WMH, Scan2Wave3_FLAIR_brain, Scan2Wave3_WMH, etc.
    Each folder contains patient .nii/.nii.gz files.
    
    Supports 3D augmentations on full volumes before slicing.
    """
    def __init__(self, root_dir, transform=None, max_slices_per_patient=None, use_wmh=False, apply_3d_augmentation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_wmh = use_wmh
        self.apply_3d_augmentation = apply_3d_augmentation
        self.index_map = []
        self.patient_ids = set()
        
        # Cache for augmented volumes to improve efficiency
        self.augmented_volume_cache = {}
        
        # Get augmentation transforms
        if self.apply_3d_augmentation:
            self.augmentation_transform = get_train_transforms()
            print("[Dataset] 3D augmentations ENABLED for training")
        else:
            self.augmentation_transform = get_val_transforms()
            print("[Dataset] 3D augmentations DISABLED")

        print(f"📂 Scanning folders in {root_dir} ...")

        # Step 1: Identify FLAIR and WMH folders by scan pair name
        folder_map = defaultdict(dict)
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            # Extract scan-pair identifier (e.g. "Scan1Wave2")
            if "FLAIR" in folder:
                key = folder.split("_FLAIR")[0]
                folder_map[key]["FLAIR"] = folder_path
            elif "WMH" in folder:
                key = folder.split("_WMH")[0]
                folder_map[key]["WMH"] = folder_path

        print(f"✅ Found {len(folder_map)} scan-pair folders (e.g. Scan1Wave2, Scan2Wave3, ...).")

        # Step 2: Iterate over all scan-pairs
        for scan_pair, paths in folder_map.items():
            if "FLAIR" not in paths:
                continue  # skip incomplete pairs
            flair_folder = paths["FLAIR"]
            wmh_folder = paths.get("WMH", None)

            # Extract numeric scan/wave indices for time delta
            import re
            match = re.match(r"Scan(\d+)Wave(\d+)", scan_pair)
            if not match:
                print(f"⚠️ Skipping malformed folder name: {scan_pair}")
                continue
            scan_idx, wave_idx = map(float, match.groups())
            time_delta = wave_idx - scan_idx

            # Step 3: Iterate over all patient files in the FLAIR folder
            for f in os.listdir(flair_folder):
                if not f.endswith((".nii", ".nii.gz")):
                    continue
                fname = os.path.basename(f)

                match = re.match(r"LBC36(\d+)_(\d+)_", fname)
                if not match:
                    print(f"⚠️ Could not parse patient info from filename: {fname}")
                    continue

                patient_id, _ = match.groups()
                flair_path = os.path.join(flair_folder, f)
                wmh_path = os.path.join(wmh_folder, f) if wmh_folder and os.path.exists(os.path.join(wmh_folder, f)) else None

                try:
                    vol = nib.load(flair_path).get_fdata(dtype=np.float32)
                    num_slices = vol.shape[2]
                    slice_indices = list(range(14, num_slices))
                    if max_slices_per_patient and len(slice_indices) > max_slices_per_patient:
                        step = len(slice_indices) / max_slices_per_patient
                        slice_indices = [int(i * step) for i in range(max_slices_per_patient)]

                    for s_idx in slice_indices:
                        self.index_map.append({
                            "patient_id": patient_id,
                            "scan_pair": scan_pair,
                            "flair_path": flair_path,
                            "wmh_path": wmh_path,
                            "slice_idx": s_idx,
                            "time_delta": time_delta
                        })
                        self.patient_ids.add(patient_id)
                except Exception as e:
                    print(f"⚠️ Could not load {flair_path}: {e}")
                    continue

        print(f"📊 Dataset ready. Found {len(self.index_map)} slices from {len(self.patient_ids)} patients.")

    def _apply_3d_augmentations(self, flair_vol, wmh_vol):
        """
        Apply 3D augmentations to full volumes using MONAI transforms.
        
        Args:
            flair_vol: 3D FLAIR volume of shape (H, W, D)
            wmh_vol: 3D WMH volume of shape (H, W, D)
            
        Returns:
            Augmented flair_vol and wmh_vol (as numpy arrays)
        """
        # Convert to format expected by MONAI: (1, H, W, D) for single channel, or (C, H, W, D)
        flair_vol_aug = flair_vol[np.newaxis, ...]  # (1, H, W, D)
        wmh_vol_aug = wmh_vol[np.newaxis, ...]      # (1, H, W, D)
        
        # Prepare data dict for augmentation
        data_dict = {
            "image": flair_vol_aug,
            "label": wmh_vol_aug,
        }
        
        # Apply augmentations
        try:
            data_dict = self.augmentation_transform(data_dict)
            # Convert MetaTensor back to numpy array
            # Use .numpy() method for MetaTensor to avoid deprecation warnings
            flair_vol_aug = data_dict["image"][0].numpy() if hasattr(data_dict["image"][0], 'numpy') else np.array(data_dict["image"][0])
            wmh_vol_aug = data_dict["label"][0].numpy() if hasattr(data_dict["label"][0], 'numpy') else np.array(data_dict["label"][0])
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}. Using original volumes.")
            flair_vol_aug = flair_vol
            wmh_vol_aug = wmh_vol
        
        return flair_vol_aug, wmh_vol_aug

    def _load_slice(self, file_path, slice_idx, apply_augmentation=False):
        """Helper to load a single slice from a 3D NIfTI file."""
        cache_key = f"{file_path}_{apply_augmentation}"
        
        # Load volume (possibly from cache if augmented)
        if cache_key not in self.augmented_volume_cache:
            vol = nib.load(file_path).get_fdata(dtype=np.float32)
            
            # Apply 3D augmentations if requested
            if apply_augmentation and self.apply_3d_augmentation:
                # For augmentation, we need both FLAIR and WMH volumes
                # This will be handled in __getitem__
                pass
            
            self.augmented_volume_cache[cache_key] = vol
        else:
            vol = self.augmented_volume_cache[cache_key]
        
        # Extract slice
        img_slice = vol[:, :, slice_idx]
        
        # Normalize slice to [0, 1] range
        if img_slice.max() - img_slice.min() > 1e-8:
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        
        return torch.from_numpy(img_slice).unsqueeze(0)  # [1, H, W]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        info = self.index_map[idx]
        flair_path = info["flair_path"]
        wmh_path = info["wmh_path"]
        s_idx = info["slice_idx"]
        
        # Load full volumes
        flair_vol = nib.load(flair_path).get_fdata(dtype=np.float32)
        if wmh_path:
            wmh_vol = nib.load(wmh_path).get_fdata(dtype=np.float32)
        else:
            wmh_vol = np.zeros_like(flair_vol)
        
        # Apply 3D augmentations if enabled
        if self.apply_3d_augmentation:
            flair_vol, wmh_vol = self._apply_3d_augmentations(flair_vol, wmh_vol)
        
        # Extract slice from (possibly augmented) volume
        flair_slice = flair_vol[:, :, s_idx]
        wmh_slice = wmh_vol[:, :, s_idx]
        
        # Normalize slices independently to [0, 1]
        if flair_slice.max() - flair_slice.min() > 1e-8:
            flair_slice = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min())
        if wmh_slice.max() - wmh_slice.min() > 1e-8:
            wmh_slice = (wmh_slice - wmh_slice.min()) / (wmh_slice.max() - wmh_slice.min())
        
        # Convert to tensors
        flair_slice = torch.from_numpy(flair_slice).unsqueeze(0)  # [1, H, W]
        wmh_slice = torch.from_numpy(wmh_slice).unsqueeze(0)      # [1, H, W]
        
        # Combine as multi-channel [FLAIR, WMH]
        source_img = torch.cat([flair_slice, wmh_slice], dim=0)  # [2, H, W]
        
        return {
            "source_image": source_img,
            "target_image": source_img,  # target = same FLAIR for now, modified later in pipeline
            "time_delta": torch.tensor(info["time_delta"], dtype=torch.float32),
            "patient_id": info["patient_id"],
            "slice_idx": torch.tensor(info["slice_idx"], dtype=torch.long)
        }

# --- Configuration ---
ROOT_DIR = "/app/dataset/LBC1936-FLAIR-WMH"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
MAX_SLICES = 48
MAX_PATIENTS_PER_FOLD = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECON_PSNR_THR = 25.0  # PSNR threshold to start training the ODE component
CONTRASTIVE_COEFF = 0.1 # Weight for the contrastive loss term
CV_FOLDS = [1]  # Folds to use for cross-validation (single fold)
TEST_FOLD = 2   # The single, held-out test fold
K_FOLDS = len(CV_FOLDS)  # The number of models we will train (will be 1)

print(f"Using device: {DEVICE}")

# --- Visualization and Testing ---

def visualize_results(source, ground_truth, predicted, patient_ids, slice_indices, n_samples=5, filename="test_results.png", save_dir="flair-to-flair-results"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(15, n_samples * 5))
    n_samples = min(n_samples, source.shape[0])

    for i in range(n_samples):
        p_id = patient_ids[i]
        s_idx = slice_indices[i].item()

        plt.subplot(n_samples, 3, i * 3 + 1)
        flair = source[i, 0].cpu().numpy()
        wmh = source[i, 1].cpu().numpy()
        plt.imshow(flair, cmap='gray')
        plt.imshow(wmh, cmap='Reds', alpha=0.4)

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

def plot_fold_history(history, fold_idx, save_dir="flair-to-flair-results"):
    """Plots and saves the training and validation history for a fold."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"training_history_fold_{fold_idx}.png")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history['train_recon_loss']) + 1)

    # Plot training losses on the primary y-axis
    ax1.plot(epochs, history['train_recon_loss'], 'b-', label='Train Recon Loss')
    ax1.plot(epochs, history['train_pred_loss'], 'g-', label='Train Pred Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(bottom=0)

    # Create a secondary y-axis for validation PSNR
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['val_recon_psnr'], 'r--', label='Val Recon PSNR')
    ax2.plot(epochs, history['val_pred_psnr'], 'm--', label='Val Pred PSNR')
    ax2.set_ylabel('Validation PSNR (dB)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add titles and legend
    fig.suptitle(f'Training History - Fold {fold_idx}')
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Training history plot saved to {save_path}")

# Unused now
def test_single_model(model_path, test_loader, scan_pair="Scan1Wave2"):
    print(f"\n--- Evaluating model: {model_path} ({scan_pair}) ---")

    # Initialize model
    model = ImageFlowNetODE(
        device=DEVICE, in_channels=2, ode_location='bottleneck', contrastive=True
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dice = DiceMetric()
    mse = torchmetrics.MeanSquaredError().to(DEVICE)
    psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    # Create output folders
    save_dirs = {
        "T1_interp": f"flair-to-flair-results/{scan_pair}_predicted_flairs_T1",
        "T2_normal": f"flair-to-flair-results/{scan_pair}_predicted_flairs_T2",
        "T3_extra":  f"flair-to-flair-results/{scan_pair}_predicted_flairs_T3",
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Testing {os.path.basename(model_path)}")):
            source_img = batch["source_image"].to(DEVICE)
            target_img = batch["target_image"].to(DEVICE)
            time_deltas = batch["time_delta"].to(DEVICE)
            slice_indices, patient_ids = batch["slice_idx"], batch["patient_id"]

            # --- Normal forward (Δt → T2)
            t_normal = time_deltas[0:1]
            pred_T2 = torch.sigmoid(model(source_img, t=t_normal))

            # --- Interpolation (0.5 × Δt → T1)
            t_half = time_deltas[0:1] * 0.5
            pred_T1 = torch.sigmoid(model(source_img, t=t_half))

            # --- Extrapolation (1.5 × Δt → T3)
            t_future = time_deltas[0:1] * 1.5
            pred_T3 = torch.sigmoid(model(source_img, t=t_future))

            # --- Metrics (using Δt → T2)
            dice.update(pred_T2, target_img)
            mse.update(pred_T2, target_img)
            psnr.update(pred_T2, target_img)
            ssim.update(pred_T2, target_img)

            # --- Save predictions as NIfTI
            for tag, preds in zip(["T1_interp", "T2_normal", "T3_extra"], [pred_T1, pred_T2, pred_T3]):
                for j in range(preds.shape[0]):
                    pred_np = preds[j, 0].cpu().numpy()
                    fname = f"{patient_ids[j]}_slice{slice_indices[j].item()}.nii.gz"
                    nib.save(nib.Nifti1Image(pred_np, np.eye(4)),
                             os.path.join(save_dirs[tag], fname))

            # --- Visualization for first batch
            if i == 0:
                visualize_results(source_img, target_img, pred_T2,
                                  patient_ids, slice_indices,
                                  filename=f"{scan_pair}_T2_normal.png")
                visualize_results(source_img, target_img, pred_T1,
                                  patient_ids, slice_indices,
                                  filename=f"{scan_pair}_T1_interp.png")
                visualize_results(source_img, target_img, pred_T3,
                                  patient_ids, slice_indices,
                                  filename=f"{scan_pair}_T3_extra.png")

    final_dice, final_mse = dice.compute().item(), mse.compute().item()
    final_psnr, final_ssim = psnr.compute().item(), ssim.compute().item()

    print(f"Results for {model_path}: "
          f"Dice={final_dice:.4f}, PSNR={final_psnr:.4f}, "
          f"SSIM={final_ssim:.4f}, MSE={final_mse:.6f}")
    print(f"Saved predictions:\n"
          f"  ↳ Interpolation (T1): {save_dirs['T1_interp']}\n"
          f"  ↳ Normal (T2):        {save_dirs['T2_normal']}\n"
          f"  ↳ Extrapolation (T3): {save_dirs['T3_extra']}\n")

    return {
        'model_path': model_path,
        'Dice': final_dice,
        'MSE': final_mse,
        'PSNR': final_psnr,
        'SSIM': final_ssim,
        'dirs': save_dirs
    }

def evaluate_and_visualize_tasks(model_path, source_loader, gt_loaders, device, original_scans_dir):
    """
    Evaluates a model on specific tasks (t=1,2,3), calculates metrics,
    collects all predicted slices, and saves them as one 3D NIfTI file per patient.
    """
    print(f"\n--- Running Full Evaluation for: {os.path.basename(model_path)} ---")

    # --- Initialize Model ---
    model = ImageFlowNetODE(device=device, in_channels=2, ode_location='bottleneck', contrastive=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Define tasks and initialize metrics ---
    tasks = {
        "Interpolation_t1": {"scan_pair": "Scan2Wave3", "time": 1.0},
        "Prediction_t2":    {"scan_pair": "Scan3Wave4", "time": 2.0},
        "Extrapolation_t3": {"scan_pair": "Scan4Wave5", "time": 3.0},
    }
    metrics = {name: {
        "psnr": torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device),
        "ssim": torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    } for name in tasks}
    
    # --- Data structure to collect predictions before saving ---
    # Format: { 'task_name': { 'patient_id': {slice_idx: slice_data} } }
    patient_predictions = {task_name: defaultdict(dict) for task_name in tasks}

    gt_iterators = {task: iter(loader) for task, loader in gt_loaders.items()}
    
    # --- Stage 1: Collect all predictions from the loader ---
    with torch.no_grad():
        for i, source_batch in enumerate(tqdm(source_loader, desc="Evaluating all tasks")):
            source_img = source_batch["source_image"].to(device)
            patient_ids, slice_indices = source_batch["patient_id"], source_batch["slice_idx"]

            for task_name, task_info in tasks.items():
                try:
                    # --- Prediction ---
                    t = torch.tensor([task_info["time"]], device=device)
                    pred_img = torch.sigmoid(model(source_img, t=t))

                    # --- Metrics ---
                    gt_pair_name = task_info["scan_pair"]
                    gt_batch = next(gt_iterators[gt_pair_name])
                    target_img = gt_batch["target_image"].to(device)
                    metrics[task_name]["psnr"].update(pred_img, target_img)
                    metrics[task_name]["ssim"].update(pred_img, target_img)
                    
                    # --- Collect slice data instead of saving immediately ---
                    for j in range(pred_img.shape[0]):
                        p_id = patient_ids[j]
                        s_idx = slice_indices[j].item()
                        pred_np = pred_img[j, 0].cpu().numpy()
                        patient_predictions[task_name][p_id][s_idx] = pred_np

                    # --- Visualization (still happens on the first batch) ---
                    if i == 0:
                        model_prefix = os.path.basename(model_path).split('.')[0]
                        visualize_results(
                            source=source_img, ground_truth=target_img, predicted=pred_img,
                            patient_ids=patient_ids, slice_indices=slice_indices,
                            filename=f"Comparison_{model_prefix}_to_{gt_pair_name}.png",
                            save_dir="flair-to-flair-results"
                        )
                except StopIteration:
                    continue

    # --- Stage 2: Save collected predictions as 3D NIfTI volumes ---
    print("\n💾 Collecting and saving predictions as 3D NIfTI volumes...")
    for task_name, predictions_by_patient in tqdm(patient_predictions.items(), desc="Saving 3D Volumes"):
        gt_pair_name = tasks[task_name]["scan_pair"]
        model_prefix = os.path.basename(model_path).split('.')[0]
        save_dir = f"flair-to-flair-results/{model_prefix}_Pred_{gt_pair_name}_3D"
        os.makedirs(save_dir, exist_ok=True)

        for patient_id, slices in predictions_by_patient.items():
            if not slices:
                continue

            # --- Determine volume dimensions and create an empty volume ---
            max_slice_idx = max(slices.keys())
            H, W = next(iter(slices.values())).shape
            volume = np.zeros((H, W, max_slice_idx + 1), dtype=np.float32)

            # --- Fill the volume with the collected slices ---
            for slice_idx, slice_data in slices.items():
                volume[:, :, slice_idx] = slice_data

            # --- Get the affine matrix from an original scan for correct orientation ---
            affine = np.eye(4) # Default identity affine
            try:
                # Find the patient's original file to borrow its spatial info
                full_prefix = f"LBC36{patient_id}"
                original_file = next(f for f in os.listdir(original_scans_dir) if f.startswith(full_prefix))
                original_nii = nib.load(os.path.join(original_scans_dir, original_file))
                affine = original_nii.affine
            except (StopIteration, FileNotFoundError):
                print(f"⚠️ Warning: Original scan for patient {patient_id} not found. Using default affine.")
            except Exception as e:
                print(f"⚠️ Warning: Could not load original scan for patient {patient_id}. Error: {e}")


            # --- Save the complete 3D NIfTI file ---
            nii_image = nib.Nifti1Image(volume, affine)
            output_filename = os.path.join(save_dir, f"{patient_id}_predicted_3D.nii.gz")
            nib.save(nii_image, output_filename)

    # --- Compile and return final results ---
    final_results = {'model_path': model_path}
    for task_name in tasks:
        final_results[task_name] = {
            "PSNR": metrics[task_name]["psnr"].compute().item(),
            "SSIM": metrics[task_name]["ssim"].compute().item(),
        }
    return final_results

def test_specific_times(model_path, source_loader, gt_loaders):
    print(f"\n--- Evaluating specific time points for model: {model_path} ---")

    # --- Initialize Model ---
    model = ImageFlowNetODE(device=DEVICE, in_channels=2, ode_location='bottleneck', contrastive=True).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # --- Define the target time points ---
    time_points = {
        "Scan2Wave3": torch.tensor([1.0], device=DEVICE), # Interpolation
        "Scan3Wave4": torch.tensor([2.0], device=DEVICE), # Normal
        "Scan4Wave5": torch.tensor([3.0], device=DEVICE), # Extrapolation
    }

    # --- Create iterators for the ground truth data ---
    gt_iterators = {name: iter(loader) for name, loader in gt_loaders.items()}

    with torch.no_grad():
        for i, source_batch in enumerate(tqdm(source_loader, desc="Predicting Future Scans")):
            source_img = source_batch["source_image"].to(DEVICE)
            patient_ids, slice_indices = source_batch["patient_id"], source_batch["slice_idx"]

            # --- Generate predictions for all target times ---
            predictions = {}
            for name, t in time_points.items():
                predictions[name] = torch.sigmoid(model(source_img, t=t))

            # --- Visualize and Save Results ---
            for name, pred_img in predictions.items():
                try:
                    # Get the corresponding ground truth batch
                    gt_batch = next(gt_iterators[name])
                    target_img = gt_batch["target_image"].to(DEVICE)

                    # Create a directory for this specific prediction type
                    save_dir = f"flair-to-flair-results/T0_to_{name}"
                    os.makedirs(save_dir, exist_ok=True)

                    # Visualize the first batch
                    if i == 0:
                        visualize_results(
                            source=source_img,
                            ground_truth=target_img,
                            predicted=pred_img,
                            patient_ids=patient_ids,
                            slice_indices=slice_indices,
                            filename=f"prediction_T0_to_{name}.png",
                            save_dir="flair-to-flair-results"
                        )

                    # Save predictions as NIfTI files
                    for j in range(pred_img.shape[0]):
                        pred_np = pred_img[j, 0].cpu().numpy()
                        # Use patient and slice info from the source batch for consistent naming
                        fname = f"{patient_ids[j]}_slice{slice_indices[j].item()}.nii.gz"
                        nib.save(nib.Nifti1Image(pred_np, np.eye(4)), os.path.join(save_dir, fname))

                except StopIteration:
                    # Ran out of ground truth images; this can happen if datasets aren't perfectly aligned
                    print(f"Warning: No more ground truth images for {name} to compare with.")
                    continue

    print("\n✅ Specific time point evaluation complete.")
    print("Predictions saved in 'flair-to-flair-results/' directory with corresponding scan names.")

# === MODIFIED: train_epoch now returns average losses ===
def train_epoch(model, loader, optimizer, ema, mse_loss, device, epoch_idx, train_time_dependent):
    model.train()

    total_recon_loss = 0.0
    total_pred_loss = 0.0
    pred_loss_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{NUM_EPOCHS} [Train]")
    for i, batch in enumerate(pbar):
        source_img, target_img = batch["source_image"].to(DEVICE), batch["target_image"].to(DEVICE)
        time_deltas = batch["time_delta"].to(DEVICE)

        optimizer.zero_grad()

        # --- 1. Reconstruction Loss (Trains Encoder/Decoder) ---
        if hasattr(model, 'unfreeze'):
            model.unfreeze()

        source_recon = model(source_img, t=torch.zeros(1).to(device))
        target_recon = model(target_img, t=torch.zeros(1).to(device))
        loss_recon = mse_loss(source_recon, source_img) + mse_loss(target_recon, target_img)

        if hasattr(model, 'simsiam_project') and hasattr(model, 'simsiam_predict'):
            z1, z2 = model.simsiam_project(source_img), model.simsiam_project(target_img)
            p1, p2 = model.simsiam_predict(z1), model.simsiam_predict(z2)
            loss_contrastive = neg_cos_sim(p1, z2)/2 + neg_cos_sim(p2, z1)/2
            loss = loss_recon + CONTRASTIVE_COEFF * loss_contrastive
        else:
            loss = loss_recon

        loss.backward()
        optimizer.step()
        ema.update()
        total_recon_loss += loss.item()

        # --- 2. Prediction Loss (Trains ODE component) ---
        if train_time_dependent:
            optimizer.zero_grad()
            if hasattr(model, 'freeze_time_independent'):
                model.freeze_time_independent()

            t = time_deltas[0:1]
            predicted_target = model(source_img, t)
            loss_pred = mse_loss(predicted_target, target_img)

            loss_pred.backward()
            optimizer.step()
            ema.update()

            total_pred_loss += loss_pred.item()
            pred_loss_count += 1

        pbar.set_postfix(
            recon_loss=total_recon_loss / (i + 1),
            pred_loss=total_pred_loss / pred_loss_count if pred_loss_count > 0 else "N/A"
        )
    
    avg_recon_loss = total_recon_loss / len(loader)
    avg_pred_loss = total_pred_loss / pred_loss_count if pred_loss_count > 0 else float('nan')
    return avg_recon_loss, avg_pred_loss


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
    """Load predefined patient folds from CSV and normalize ID format."""
    df = pd.read_csv(fold_csv_path, dtype={"patient_ID": str})  
    folds = {}
    for _, row in df.iterrows():
        pid = str(row["patient_ID"])
        # normalize to numeric-only form, e.g. "LBC0375" -> "0375"
        pid = re.sub(r"^LBC", "", pid)
        pid = re.sub(r"^LBC36", "", pid)  # also handles "LBC36XXXX"
        pid = pid.zfill(4)                
        fold = int(row["fold"])
        folds.setdefault(fold, []).append(pid)
    return folds

# Main logic
def main():
    print("Initializing dataset...")
    # Create training dataset WITH 3D augmentations
    train_dataset = FLAIREvolutionDataset(
        root_dir=ROOT_DIR, 
        max_slices_per_patient=MAX_SLICES,
        apply_3d_augmentation=True  # Enable augmentations for training
    )
    
    # Create validation dataset WITHOUT 3D augmentations
    val_dataset = FLAIREvolutionDataset(
        root_dir=ROOT_DIR, 
        max_slices_per_patient=MAX_SLICES,
        apply_3d_augmentation=False  # Disable augmentations for validation
    )

    # Load fold assignments from CSV
    fold_csv = "patients_5fold.csv"
    if not os.path.exists(fold_csv):
        raise FileNotFoundError(f"Fold CSV not found at {fold_csv}. Please provide it.")
    folds_dict = load_folds_from_csv(fold_csv)
    print(f"Loaded patient folds from {fold_csv}.")
    print(f"Example dataset patient IDs: {list(train_dataset.patient_ids)[:5]}")

    # --- 1. K-Fold Cross-Validation Training ---
    print(f"\n" + "="*50 + f"\n📈 Starting {K_FOLDS}-Fold Cross-Validation Training..." + "\n" + "="*50)

    for val_fold_idx in CV_FOLDS:
        print(f"\n" + "="*50 + f"\n K-Fold Run: Validating on Fold {val_fold_idx} " + "\n" + "="*50)

        # --- Define splits for this specific run ---
        if K_FOLDS == 1:
            # For single fold, split the fold into train (80%) and validation (20%)
            fold_pids = folds_dict[val_fold_idx][:MAX_PATIENTS_PER_FOLD]
            split_point = int(0.8 * len(fold_pids))
            train_pids = fold_pids[:split_point]
            val_pids = fold_pids[split_point:]
            print(f"[Single Fold Mode] Splitting fold {val_fold_idx} into train/val (80/20)")
        else:
            # Original multi-fold logic: validate on one fold, train on others
            val_pids = folds_dict[val_fold_idx][:MAX_PATIENTS_PER_FOLD]
            train_pids = [
                pid for f_idx in CV_FOLDS if f_idx != val_fold_idx 
                for pid in folds_dict[f_idx]
            ][:MAX_PATIENTS_PER_FOLD * (K_FOLDS - 1)]
        
        print(f"Example train_pids (Fold {val_fold_idx}): {train_pids[:5]}")
        print(f"Training patients:   {len(train_pids)}")
        print(f"Validation patients: {len(val_pids)}")

        # Create dataset indices
        train_indices = [i for i, item in enumerate(train_dataset.index_map) if item['patient_id'] in set(train_pids)]
        val_indices   = [i for i, item in enumerate(val_dataset.index_map) if item['patient_id'] in set(val_pids)]

        # Create DataLoaders
        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader   = DataLoader(Subset(val_dataset, val_indices),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # --- Initialize a new model for this fold ---
        model = ImageFlowNetODE(device=DEVICE, in_channels=2, ode_location='bottleneck', contrastive=True).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=NUM_EPOCHS // 10, max_epochs=NUM_EPOCHS)
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
        mse_loss = nn.MSELoss()

        best_val_psnr = 0.0
        recon_good_enough = False
        model_save_path = f"flair_to_flair_model_fold_{val_fold_idx}.pth"
        
        history = {'train_recon_loss': [], 'train_pred_loss': [], 'val_recon_psnr': [], 'val_pred_psnr': []}

        for epoch in range(NUM_EPOCHS):
            avg_recon_loss, avg_pred_loss = train_epoch(model, train_loader, optimizer, ema, mse_loss, DEVICE, epoch, recon_good_enough)
            with ema.average_parameters():
                val_recon_psnr, val_pred_psnr = val_epoch(model, val_loader, DEVICE)
            
            history['train_recon_loss'].append(avg_recon_loss)
            history['train_pred_loss'].append(avg_pred_loss)
            history['val_recon_psnr'].append(val_recon_psnr)
            history['val_pred_psnr'].append(val_pred_psnr)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} (Fold {val_fold_idx}): Val PSNR={val_pred_psnr:.4f}")

            if not recon_good_enough and val_recon_psnr > RECON_PSNR_THR:
                recon_good_enough = True
                print("Reconstruction threshold reached. Starting ODE training.")

            if val_pred_psnr > best_val_psnr:
                best_val_psnr = val_pred_psnr
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Val PSNR improved. Model saved to {model_save_path}")

            scheduler.step()
        
        plot_fold_history(history, val_fold_idx)

    # --- 2. Final Evaluation on the Held-Out Test Set ---
    print("\n" + "="*60 + "\n✅ CV Training Complete. Starting Final Evaluation on Held-Out Test Set." + "\n" + "="*60)
    
    # Define the single, held-out test set (using val_dataset without augmentation)
    test_pids = folds_dict[TEST_FOLD][:MAX_PATIENTS_PER_FOLD]
    test_indices = [i for i, item in enumerate(val_dataset.index_map) if item['patient_id'] in set(test_pids)]
    print(f"Using {len(test_pids)} patients from Fold {TEST_FOLD} for final testing.")

    # Create DataLoaders for the test set
    source_loader = DataLoader(Subset(val_dataset, [i for i in test_indices if val_dataset.index_map[i]['scan_pair'] == "Scan1Wave2"]), batch_size=BATCH_SIZE, shuffle=False)
    gt_loaders = {
        "Scan2Wave3": DataLoader(Subset(val_dataset, [i for i in test_indices if val_dataset.index_map[i]['scan_pair'] == "Scan2Wave3"]), batch_size=BATCH_SIZE, shuffle=False),
        "Scan3Wave4": DataLoader(Subset(val_dataset, [i for i in test_indices if val_dataset.index_map[i]['scan_pair'] == "Scan3Wave4"]), batch_size=BATCH_SIZE, shuffle=False),
        "Scan4Wave5": DataLoader(Subset(val_dataset, [i for i in test_indices if val_dataset.index_map[i]['scan_pair'] == "Scan4Wave5"]), batch_size=BATCH_SIZE, shuffle=False),
    }

    # Evaluate ALL models trained during cross-validation
    model_paths = [f"flair_to_flair_model_fold_{i}.pth" for i in CV_FOLDS if os.path.exists(f"flair_to_flair_model_fold_{i}.pth")]
    if not model_paths:
        print("No trained models found to evaluate.")
        return None, None

    original_scans_dir_for_affine = os.path.join(ROOT_DIR, "Scan1Wave2_FLAIR_brain")
    all_results = [evaluate_and_visualize_tasks(path, source_loader, gt_loaders, DEVICE, original_scans_dir=original_scans_dir_for_affine) for path in model_paths]

    # --- 3. Report Final Aggregated Results ---
    print("\n" + "="*60 + "\n============= Final Test Set Results (Mean +/- Std Dev) =============" + "\n" + "="*60)
    
    # Aggregate metrics from all fold models
    interp_psnrs = [r['Interpolation_t1']['PSNR'] for r in all_results]
    pred_psnrs   = [r['Prediction_t2']['PSNR'] for r in all_results]
    extrap_psnrs = [r['Extrapolation_t3']['PSNR'] for r in all_results]

    print(f"Interpolation PSNR (t=1->2): {np.mean(interp_psnrs):.4f} +/- {np.std(interp_psnrs):.4f}")
    print(f"Prediction PSNR    (t=1->3): {np.mean(pred_psnrs):.4f} +/- {np.std(pred_psnrs):.4f}")
    print(f"Extrapolation PSNR (t=1->4): {np.mean(extrap_psnrs):.4f} +/- {np.std(extrap_psnrs):.4f}")
    print("="*60)

    # Find the single best model based on prediction PSNR to use for Stage 2
    best_result = max(all_results, key=lambda x: x['Prediction_t2']['PSNR'])
    best_model_name = os.path.basename(best_result['model_path']).split('.')[0]
    print(f"🏆 Best single model for Stage 2: {best_model_name} (Prediction PSNR: {max(pred_psnrs):.4f})")

    predicted_flair_dir = f"flair-to-flair-results/{best_model_name}_Pred_Scan3Wave4"
    ground_truth_wmh_dir = os.path.join(ROOT_DIR, "Scan3Wave4_WMH")

    return predicted_flair_dir, ground_truth_wmh_dir

# ============================================================
# === STAGE 2 - WMH SEGMENTATION FROM PREDICTED FLAIR ===
# ============================================================

def save_segmentation_sample(model, loader, device, filename="segmentation_sample.png"):
    """
    Saves a side-by-side visualization of one sample:
    [Input FLAIR | Ground Truth Mask | Predicted Mask]
    """
    print(f"\n[Visualizing] Saving a random sample to '{filename}'...")
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        # Get a single batch from the dataloader
        sample_batch = next(iter(loader))
        x, y = sample_batch["flair"].to(device), sample_batch["mask"].to(device)
        
        # Get the model's prediction
        p = model(x)
        
        # --- Prepare tensors for plotting ---
        # Select the very first item from the batch
        # Move to CPU, remove channel dimension, convert to NumPy
        flair_img = x[0].cpu().squeeze().numpy()
        true_mask = y[0].cpu().squeeze().numpy()
        
        # Binarize the prediction for clear visualization (threshold at 0.5)
        pred_mask = (p[0].cpu().squeeze().numpy() > 0.5).astype(float)

        # --- Create the plot ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot Input FLAIR
        axes[0].imshow(flair_img, cmap='gray')
        axes[0].set_title("Input (Predicted FLAIR)")
        axes[0].axis('off')
        
        # Plot Ground Truth WMH Mask
        axes[1].imshow(true_mask, cmap='jet')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

        # Plot Predicted WMH Mask
        axes[2].imshow(pred_mask, cmap='jet')
        axes[2].set_title("Swin UNETR Predicted Mask")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"[Visualizing] Sample saved successfully.")

class DownstreamSegmentationDataset(Dataset):
    """
    Loads 2D slices from corresponding 3D predicted FLAIR and 3D ground truth WMH volumes.
    This version assumes the data is in a consistent 3D format.
    """
    def __init__(self, pred_flair_dir_3d, wmh_gt_dir):
        self.index_map = []
        
        # Find matching 3D file pairs
        print(f"Matching 3D volumes between '{pred_flair_dir_3d}' and '{wmh_gt_dir}'...")
        for pred_file in os.listdir(pred_flair_dir_3d):
            # CORRECTED: Parse the numeric ID from predicted filenames like '0123_predicted_3D.nii.gz'
            match = re.match(r"(\d+)_predicted_3D\.nii\.gz", pred_file)
            if not match:
                continue
            
            # This is the numeric part of the ID, e.g., '0123'
            patient_id_num = match.group(1) 
            # Ground truth files start with a prefix like 'LBC360123'
            patient_gt_prefix = f"LBC36{patient_id_num}"
            
            found_gt_file = None
            
            # Find the corresponding ground truth file for this patient using the constructed prefix
            for gt_file in os.listdir(wmh_gt_dir):
                if gt_file.startswith(patient_gt_prefix):
                    found_gt_file = gt_file
                    break
            
            if found_gt_file:
                pred_path = os.path.join(pred_flair_dir_3d, pred_file)
                gt_path = os.path.join(wmh_gt_dir, found_gt_file)
                
                try:
                    # Get the number of slices to create our index
                    num_slices = nib.load(pred_path).shape[2]
                    for s_idx in range(num_slices):
                        self.index_map.append({
                            "pred_path": pred_path,
                            "gt_path": gt_path,
                            "slice_idx": s_idx
                        })
                except Exception as e:
                    print(f"⚠️ Could not process pair: {pred_file} and {found_gt_file}. Error: {e}")

        if not self.index_map:
            print("⚠️ WARNING: No matching predicted FLAIR and ground truth WMH volumes were found. The dataset is empty.")

        print(f"[DownstreamDataset] Found {len(self.index_map)} total slices from matched 3D volumes.")

    def __len__(self):
        return len(self.index_map)

    def _load_slice(self, file_path, slice_idx):
        """Helper to load a single slice from a 3D NIfTI file."""
        vol = nib.load(file_path).get_fdata(dtype=np.float32)
        img_slice = vol[:, :, slice_idx]
        # Normalize slice to [0, 1] range
        if img_slice.max() - img_slice.min() > 1e-8:
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        return torch.from_numpy(img_slice).unsqueeze(0)

    def __getitem__(self, idx):
        info = self.index_map[idx]
        s_idx = info["slice_idx"]
        
        # Load the corresponding slices from the 3D predicted FLAIR and GT WMH volumes
        flair_slice = self._load_slice(info["pred_path"], s_idx)
        mask_slice = self._load_slice(info["gt_path"], s_idx)
        
        return {"flair": flair_slice, "mask": mask_slice}


class UNetSegmentation(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU())
        self.e1 = block(in_ch, 32);  self.p1 = nn.MaxPool2d(2)
        self.e2 = block(32, 64);   self.p2 = nn.MaxPool2d(2)
        self.b  = block(64, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = block(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = block(64, 32)
        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        b  = self.b(self.p2(e2))
        d2 = self.u2(b)
        d2 = self.d2(torch.cat([d2, e2], 1))
        d1 = self.u1(d2)
        d1 = self.d1(torch.cat([d1, e1], 1))
        return torch.sigmoid(self.out(d1))


class SwinUNetSegmentation(nn.Module):
    """
    2D Medical Image Segmentation using MONAI's SwinUNETR.
    
    This wrapper adapts the 3D SwinUNETR for 2D slice-based processing.
    The model uses a pretrained Swin Transformer backbone with a U-shaped decoder.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale FLAIR)
        out_channels: Number of output channels (default: 1 for binary WMH mask)
        img_size: Expected input image size (default: 256x256) - for reference only
        feature_size: Number of feature channels in the first layer (default: 48)
        use_checkpoint: Whether to use gradient checkpointing to save memory
    """
    def __init__(self, in_channels=1, out_channels=1, img_size=256, feature_size=48, use_checkpoint=False):
        super().__init__()
        
        # Initialize MONAI SwinUNETR model
        # Note: SwinUNETR in MONAI for 2D requires spatial_dims=2
        # The model will automatically handle input tensor shapes
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=2,  # Use 2D for 2D slice-based processing
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Segmentation mask of shape (B, 1, H, W) with sigmoid activation
        """
        # Forward pass through SwinUNETR
        logits = self.model(x)
        
        # Apply sigmoid activation for binary segmentation
        return torch.sigmoid(logits)


def dice_loss(pred, gt, eps=1e-6):
    p, g = pred.view(-1), gt.view(-1)
    inter = (p * g).sum()
    return 1 - (2*inter + eps) / (p.sum() + g.sum() + eps)


def compute_wmh_volume(mask, voxel_volume=1.0):
    mask_bin = (mask > 0.5).float()
    return (mask_bin.sum(dim=[1,2,3]) * voxel_volume).cpu().numpy()


def train_segmentation(model, loader, opt, device):
    model.train(); tot = 0
    for b in tqdm(loader, desc="[Stage2 Train]"):
        x, y = b["flair"].to(device), b["mask"].to(device)
        p = model(x)
        loss = dice_loss(p, y) + nn.functional.binary_cross_entropy(p, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    return tot / len(loader)


def run_stage2_segmentation(pred_flair_dir, wmh_gt_dir, device):
    ds = DownstreamSegmentationDataset(pred_flair_dir, wmh_gt_dir)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    model = SwinUNetSegmentation().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for ep in range(10):
        loss = train_segmentation(model, dl, opt, device)
        print(f"[Stage2] Epoch {ep+1}: loss={loss:.4f}")
    torch.save(model.state_dict(), "wmh_segmentation_swin_unet.pth")
    print("[Stage2] Segmentation model saved as wmh_segmentation_swin_unet.pth")

    # --- Evaluate basic volume progression ---
    model.eval()
    vols_pred, vols_true, dices = [], [], []
    with torch.no_grad():
        for b in tqdm(dl, desc="[Stage2 Eval]"):
            x, y = b["flair"].to(device), b["mask"].to(device)
            p = model(x)
            dices.append(1 - dice_loss(p, y).item())
            vols_pred += list(compute_wmh_volume(p))
            vols_true += list(compute_wmh_volume(y))
    vols_pred, vols_true = np.array(vols_pred), np.array(vols_true)
    print(f"[Stage2] Mean Dice={np.mean(dices):.3f} | Delta Volume={(vols_pred - vols_true).mean():.2f}+/-{(vols_pred - vols_true).std():.2f}")

    save_segmentation_sample(model, dl, device)
    print(f"[Stage2] Segmentation sample saved to segmentation_sample.png")


def segment_3d_volume(model, volume_3d, device):
    """
    Segment a 3D volume slice by slice using the trained segmentation model.
    """
    model.eval()
    segmented_volume = np.zeros_like(volume_3d)
    
    with torch.no_grad():
        # Process each slice individually
        for slice_idx in range(volume_3d.shape[2]):
            # Extract and preprocess the slice
            slice_data = volume_3d[:, :, slice_idx]
            if slice_data.max() - slice_data.min() > 1e-8:
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            
            # Convert to tensor and add batch/channel dimensions
            slice_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Segment the slice
            pred_mask = model(slice_tensor)
            pred_mask_binary = (pred_mask > 0.5).float()
            
            # Store the result
            segmented_volume[:, :, slice_idx] = pred_mask_binary.squeeze().cpu().numpy()
    
    return segmented_volume

def calculate_volume_ml(mask_volume, voxel_size_mm=(1.0, 1.0, 1.0)):
    """
    Calculate volume in milliliters from a binary mask volume.
    Assumes voxel dimensions are in mm.
    """
    # Calculate volume in mm³
    voxel_volume_mm3 = voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]
    volume_mm3 = np.sum(mask_volume) * voxel_volume_mm3
    
    # Convert to ml (1 ml = 1000 mm³)
    volume_ml = volume_mm3 / 1000.0
    
    return volume_ml

def get_ground_truth_wmh_volume(wmh_dir, patient_id):
    """
    Load ground truth WMH volume for a specific patient.
    """
    # Construct the expected filename pattern
    patient_pattern = f"LBC36{patient_id}*.nii.gz"
    
    # Find matching files
    matching_files = []
    for file in os.listdir(wmh_dir):
        if file.startswith(f"LBC36{patient_id}") and file.endswith('.nii.gz'):
            matching_files.append(file)
    
    if not matching_files:
        print(f"⚠️ No ground truth WMH found for patient {patient_id} in {wmh_dir}")
        return None
    
    # Use the first matching file
    wmh_path = os.path.join(wmh_dir, matching_files[0])
    
    try:
        wmh_volume = nib.load(wmh_path).get_fdata(dtype=np.float32)
        return wmh_volume
    except Exception as e:
        print(f"⚠️ Could not load ground truth WMH for patient {patient_id}: {e}")
        return None

def analyze_wmh_volume_progression(predicted_flair_base_dir, gt_wmh_dirs, time_points, device):
    """
    Analyze WMH volume progression across different time points
    
    predicted_flair_base_dir: Base directory containing model predictions (e.g., 'flair-to-flair-results')
    gt_wmh_dirs: Dictionary mapping time points to ground truth WMH directories
    time_points: List of time points to analyze (e.g., ['Scan1Wave2', 'Scan2Wave3', ...])
    """
    
    volume_results = {}
    
    # Load segmentation model (trained in Stage 2)
    seg_model = SwinUNetSegmentation().to(device)
    if os.path.exists("wmh_segmentation_swin_unet.pth"):
        seg_model.load_state_dict(torch.load("wmh_segmentation_swin_unet.pth", map_location=device))
        print("✅ Loaded trained segmentation model for volume analysis")
    else:
        print("⚠️ No trained segmentation model found. Using untrained model for volume analysis.")
    
    seg_model.eval()
    
    # Construct directory paths for each time point
    # Pattern: flair_to_flair_model_fold_[fold]_Pred_[scan]_3D
    pred_dirs_by_timepoint = {}
    for time_point in time_points:
        # Find directories matching the pattern for this time point
        for folder in os.listdir(predicted_flair_base_dir):
            folder_path = os.path.join(predicted_flair_base_dir, folder)
            if os.path.isdir(folder_path) and f"Pred_{time_point}_3D" in folder:
                pred_dirs_by_timepoint[time_point] = folder_path
                print(f"  Found predictions for {time_point}: {folder}")
                break
    
    if not pred_dirs_by_timepoint:
        print("⚠️ No predicted FLAIR directories found for the specified time points.")
        return volume_results
    
    # Identify all unique patients across all time point directories
    all_patients = set()
    for time_point, pred_dir in pred_dirs_by_timepoint.items():
        for pred_file in os.listdir(pred_dir):
            if pred_file.endswith('_predicted_3D.nii.gz'):
                match = re.match(r"(\d+)_predicted_3D\.nii\.gz", pred_file)
                if match:
                    all_patients.add(match.group(1))
    
    print(f"Found {len(all_patients)} patients across all time points\n")
    
    # Process each patient
    for patient_id in all_patients:
        print(f"📊 Analyzing WMH volume progression for patient {patient_id}")
        
        patient_volumes = {'predicted': [], 'ground_truth': [], 'time_points': []}
        
        # Process each time point
        for time_point in time_points:
            if time_point not in pred_dirs_by_timepoint:
                print(f"   ⚠️ {time_point}: No directory found, skipping")
                continue
            
            pred_dir = pred_dirs_by_timepoint[time_point]
            pred_file = f"{patient_id}_predicted_3D.nii.gz"
            pred_flair_path = os.path.join(pred_dir, pred_file)
            
            # Check if file exists for this patient at this time point
            if not os.path.exists(pred_flair_path):
                print(f"   ⚠️ {time_point}: File not found for patient {patient_id}, skipping")
                continue
            
            try:
                pred_flair_volume = nib.load(pred_flair_path).get_fdata(dtype=np.float32)
                
                # Segment WMH from predicted FLAIR
                pred_wmh_volume = segment_3d_volume(seg_model, pred_flair_volume, device)
                pred_wmh_ml = calculate_volume_ml(pred_wmh_volume)
                
                # 2. Get ground truth WMH for this time point
                gt_wmh_volume = get_ground_truth_wmh_volume(gt_wmh_dirs[time_point], patient_id)
                gt_wmh_ml = calculate_volume_ml(gt_wmh_volume) if gt_wmh_volume is not None else 0
                
                patient_volumes['predicted'].append(pred_wmh_ml)
                patient_volumes['ground_truth'].append(gt_wmh_ml)
                patient_volumes['time_points'].append(time_point)
                
                print(f"   {time_point}: Predicted={pred_wmh_ml:.2f}ml, Ground Truth={gt_wmh_ml:.2f}ml")
                
            except Exception as e:
                print(f"⚠️ Error processing {time_point} for patient {patient_id}: {e}")
                continue
        
        if patient_volumes['predicted']:  # Only add if we found at least one time point
            volume_results[patient_id] = patient_volumes
        else:
            print(f"   ⚠️ No valid predictions found for patient {patient_id}\n")
    
    return volume_results

def plot_volume_progression(volume_results, save_path="volume_progression.png"):
    """Plot WMH volume progression for all patients"""
    if not volume_results:
        print("⚠️ No volume results to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    for patient_id, volumes in volume_results.items():
        time_points = volumes['time_points']
        pred_volumes = volumes['predicted']
        gt_volumes = volumes['ground_truth']
        
        # Convert time points to numeric values for plotting
        time_numeric = [i for i in range(len(time_points))]
        
        plt.subplot(2, 1, 1)
        plt.plot(time_numeric, pred_volumes, 'o-', label=f'Patient {patient_id} (Predicted)')
        plt.subplot(2, 1, 2) 
        plt.plot(time_numeric, gt_volumes, 's-', label=f'Patient {patient_id} (Ground Truth)')
    
    plt.subplot(2, 1, 1)
    plt.title('Predicted WMH Volume Progression')
    plt.xlabel('Time Point')
    plt.ylabel('WMH Volume (ml)')
    plt.xticks(range(len(time_points)), time_points, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.title('Ground Truth WMH Volume Progression') 
    plt.xlabel('Time Point')
    plt.ylabel('WMH Volume (ml)')
    plt.xticks(range(len(time_points)), time_points, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Volume progression plot saved to {save_path}")

# ============================================================
# === AUTO-LAUNCH STAGE 2 AFTER IMAGEFLOWNET TRAINING ========
# ============================================================
if __name__ == '__main__':
    # Stage 1 runs and returns the base directory name for the best predictions
    best_pred_base_dir, best_gt_dir = main()

    # Check if Stage 1 ran successfully
    if best_pred_base_dir and best_gt_dir:
        print("="*60)
        print("Starting Stage 2 (WMH Segmentation) on best model's 3D output...")
        print("="*60)

        predicted_flair_dir_3d = f"{best_pred_base_dir}_3D"
        
        print(f"[Stage 2] Using predicted 3D FLAIRs from: {predicted_flair_dir_3d}")
        print(f"[Stage 2] Using ground truth 3D WMH from: {best_gt_dir}")

        if os.path.exists(predicted_flair_dir_3d) and os.path.exists(best_gt_dir):
            # Run Stage 2 segmentation first
            run_stage2_segmentation(predicted_flair_dir_3d, best_gt_dir, DEVICE)
            
            # Then perform volumetric progression analysis
            print("="*60)
            print("Performing WMH Volume Progression Analysis")
            print("="*60)
            
            time_points_to_analyze = ['Scan1Wave2', 'Scan2Wave3', 'Scan3Wave4', 'Scan4Wave5']
            gt_wmh_dirs = {
                'Scan1Wave2': os.path.join(ROOT_DIR, "Scan1Wave2_WMH"),
                'Scan2Wave3': os.path.join(ROOT_DIR, "Scan2Wave3_WMH"), 
                'Scan3Wave4': os.path.join(ROOT_DIR, "Scan3Wave4_WMH"),
                'Scan4Wave5': os.path.join(ROOT_DIR, "Scan4Wave5_WMH")
            }
            
            # Check if all required directories exist
            missing_dirs = []
            for time_point, dir_path in gt_wmh_dirs.items():
                if not os.path.exists(dir_path):
                    missing_dirs.append(f"{time_point}: {dir_path}")
            
            if missing_dirs:
                print("⚠️ Missing ground truth directories:")
                for missing in missing_dirs:
                    print(f"   - {missing}")
                print("Skipping volumetric analysis.")
            else:
                # Pass the base results directory instead of the specific _3D directory
                volume_results = analyze_wmh_volume_progression(
                    "flair-to-flair-results", 
                    gt_wmh_dirs, 
                    time_points_to_analyze,
                    DEVICE
                )
                
                if volume_results:
                    # Plot the results
                    plot_volume_progression(volume_results)
                    
                    # Save quantitative results to CSV
                    df_results = []
                    for patient_id, volumes in volume_results.items():
                        for i, time_point in enumerate(volumes['time_points']):
                            df_results.append({
                                'patient_id': patient_id,
                                'time_point': time_point,
                                'predicted_wmh_ml': volumes['predicted'][i],
                                'ground_truth_wmh_ml': volumes['ground_truth'][i],
                                'volume_error_ml': volumes['predicted'][i] - volumes['ground_truth'][i]
                            })
                    
                    df = pd.DataFrame(df_results)
                    df.to_csv('wmh_flair_to_flair_volume_progression_results.csv', index=False)
                    print("✅ Volume progression results saved to wmh_flair_to_flair_volume_progression_results.csv")

                    # Print summary statistics
                    errors = [row['volume_error_ml'] for row in df_results]
                    if errors:
                        print(f"📊 Volume Analysis Summary:")
                        print(f"   Mean Error: {np.mean(errors):.2f} +/- {np.std(errors):.2f} ml")
                        print(f"   Min Error: {np.min(errors):.2f} ml")
                        print(f"   Max Error: {np.max(errors):.2f} ml")
                else:
                    print("⚠️ No volume results generated")
        else:
            print("[Stage 2] Skipped. Could not find the required 3D prediction directory or GT directory.")
            if not os.path.exists(predicted_flair_dir_3d):
                print(f"  -> Missing: {predicted_flair_dir_3d}")
            if not os.path.exists(best_gt_dir):
                 print(f"  -> Missing: {best_gt_dir}")
    else:
        print("[Stage 2] Skipped because Stage 1 did not complete successfully.")
