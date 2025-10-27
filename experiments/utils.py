# -*- coding: utf-8 -*-
"""
Common utilities, classes, and functions shared across all experiments.
"""

import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from torch_ema import ExponentialMovingAverage

from ImageFlowNet.src.nn.imageflownet_ode import ImageFlowNetODE
from monai.networks.nets import SwinUNETR


# ============================================================
# === LEARNING RATE SCHEDULER ===
# ============================================================

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


# ============================================================
# === LOSS AND METRIC FUNCTIONS ===
# ============================================================

def neg_cos_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity for SimSiam."""
    z = z.detach()  # Stop gradient
    p = torch.nn.functional.normalize(p, p=2, dim=1)  # L2-normalize
    z = torch.nn.functional.normalize(z, p=2, dim=1)  # L2-normalize
    return -(p * z).sum(dim=1).mean()


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


def dice_loss(pred, gt, eps=1e-6):
    p, g = pred.view(-1), gt.view(-1)
    inter = (p * g).sum()
    return 1 - (2*inter + eps) / (p.sum() + g.sum() + eps)


# ============================================================
# === DATASET CLASSES ===
# ============================================================

class FLAIREvolutionDataset(Dataset):
    """
    Custom dataset for LBC1936 folder structure where each folder encodes a scan pair:
    e.g., Scan1Wave2_FLAIR_brain, Scan1Wave2_WMH, Scan2Wave3_FLAIR_brain, etc.
    """
    def __init__(self, root_dir, transform=None, max_slices_per_patient=None, use_wmh=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_wmh = use_wmh
        self.index_map = []
        self.patient_ids = set()

        print(f"📂 Scanning folders in {root_dir} ...")

        # Step 1: Identify FLAIR and WMH folders by scan pair name
        folder_map = defaultdict(dict)
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            if "FLAIR" in folder:
                key = folder.split("_FLAIR")[0]
                folder_map[key]["FLAIR"] = folder_path
            elif "WMH" in folder:
                key = folder.split("_WMH")[0]
                folder_map[key]["WMH"] = folder_path

        print(f"✅ Found {len(folder_map)} scan-pair folders.")

        # Step 2: Iterate over all scan-pairs
        for scan_pair, paths in folder_map.items():
            if "FLAIR" not in paths:
                continue
            flair_folder = paths["FLAIR"]
            wmh_folder = paths.get("WMH", None)

            # Extract numeric scan/wave indices for time delta
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
        print(f"🔧 Configuration: use_wmh = {self.use_wmh}")

    def _load_slice(self, file_path, slice_idx):
        vol = nib.load(file_path).get_fdata(dtype=np.float32)
        img_slice = vol[:, :, slice_idx]
        if img_slice.max() - img_slice.min() > 1e-8:
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        return torch.from_numpy(img_slice).unsqueeze(0)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        info = self.index_map[idx]
        flair_slice = self._load_slice(info["flair_path"], info["slice_idx"])
        if self.use_wmh and info["wmh_path"]:
            wmh_slice = self._load_slice(info["wmh_path"], info["slice_idx"])
        else:
            wmh_slice = torch.zeros_like(flair_slice)
        source_img = torch.cat([flair_slice, wmh_slice], dim=0)
        return {
            "source_image": source_img,
            "target_image": source_img,
            "time_delta": torch.tensor(info["time_delta"], dtype=torch.float32),
            "patient_id": info["patient_id"],
            "slice_idx": torch.tensor(info["slice_idx"], dtype=torch.long)
        }


class DownstreamSegmentationDataset(Dataset):
    """Loads 2D slices from corresponding 3D predicted FLAIR and 3D ground truth WMH volumes."""
    def __init__(self, pred_flair_dir_3d, wmh_gt_dir):
        self.index_map = []
        
        print(f"Matching 3D volumes between '{pred_flair_dir_3d}' and '{wmh_gt_dir}'...")
        for pred_file in os.listdir(pred_flair_dir_3d):
            match = re.match(r"(\d+)_predicted_3D\.nii\.gz", pred_file)
            if not match:
                continue
            
            patient_id_num = match.group(1)
            patient_gt_prefix = f"LBC36{patient_id_num}"
            
            found_gt_file = None
            for gt_file in os.listdir(wmh_gt_dir):
                if gt_file.startswith(patient_gt_prefix):
                    found_gt_file = gt_file
                    break
            
            if found_gt_file:
                pred_path = os.path.join(pred_flair_dir_3d, pred_file)
                gt_path = os.path.join(wmh_gt_dir, found_gt_file)
                
                try:
                    num_slices = nib.load(pred_path).shape[2]
                    for s_idx in range(num_slices):
                        self.index_map.append({
                            "pred_path": pred_path,
                            "gt_path": gt_path,
                            "slice_idx": s_idx
                        })
                except Exception as e:
                    print(f"⚠️ Could not process pair: {pred_file} and {found_gt_file}. Error: {e}")

        print(f"[DownstreamDataset] Found {len(self.index_map)} total slices.")

    def __len__(self):
        return len(self.index_map)

    def _load_slice(self, file_path, slice_idx):
        vol = nib.load(file_path).get_fdata(dtype=np.float32)
        img_slice = vol[:, :, slice_idx]
        if img_slice.max() - img_slice.min() > 1e-8:
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        return torch.from_numpy(img_slice).unsqueeze(0)

    def __getitem__(self, idx):
        info = self.index_map[idx]
        s_idx = info["slice_idx"]
        flair_slice = self._load_slice(info["pred_path"], s_idx)
        mask_slice = self._load_slice(info["gt_path"], s_idx)
        return {"flair": flair_slice, "mask": mask_slice}


# ============================================================
# === SEGMENTATION MODEL ===
# ============================================================

class SwinUNetSegmentation(nn.Module):
    """2D Medical Image Segmentation using MONAI's SwinUNETR."""
    def __init__(self, in_channels=1, out_channels=1, img_size=256, feature_size=48, use_checkpoint=False):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=2,
        )
        
    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)


# ============================================================
# === TRAINING AND VALIDATION FUNCTIONS ===
# ============================================================

def train_epoch(model, loader, optimizer, ema, mse_loss, device, epoch_idx, train_time_dependent, num_epochs, contrastive_coeff):
    model.train()

    total_recon_loss = 0.0
    total_pred_loss = 0.0
    pred_loss_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{num_epochs} [Train]")
    for i, batch in enumerate(pbar):
        source_img, target_img = batch["source_image"].to(device), batch["target_image"].to(device)
        time_deltas = batch["time_delta"].to(device)

        optimizer.zero_grad()

        # Reconstruction Loss
        if hasattr(model, 'unfreeze'):
            model.unfreeze()

        source_recon = model(source_img, t=torch.zeros(1).to(device))
        target_recon = model(target_img, t=torch.zeros(1).to(device))
        loss_recon = mse_loss(source_recon, source_img) + mse_loss(target_recon, target_img)

        if hasattr(model, 'simsiam_project') and hasattr(model, 'simsiam_predict'):
            z1, z2 = model.simsiam_project(source_img), model.simsiam_project(target_img)
            p1, p2 = model.simsiam_predict(z1), model.simsiam_predict(z2)
            loss_contrastive = neg_cos_sim(p1, z2)/2 + neg_cos_sim(p2, z1)/2
            loss = loss_recon + contrastive_coeff * loss_contrastive
        else:
            loss = loss_recon

        loss.backward()
        optimizer.step()
        ema.update()
        total_recon_loss += loss.item()

        # Prediction Loss
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

    recon_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    pred_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)

    with torch.no_grad():
        for batch in loader:
            source_img, target_img = batch["source_image"].to(device), batch["target_image"].to(device)
            time_deltas = batch["time_delta"].to(device)

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


# ============================================================
# === VISUALIZATION FUNCTIONS ===
# ============================================================

def visualize_results(source, ground_truth, predicted, patient_ids, slice_indices, n_samples=5, filename="test_results.png", save_dir="results"):
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
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(n_samples, 3, i * 3 + 3)
        plt.imshow(predicted[i, 0].cpu().numpy(), cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visual results saved to {save_path}")


def plot_fold_history(history, fold_idx, save_dir="plots"):
    """Plots and saves the training and validation history for a fold."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"training_history_fold_{fold_idx}.png")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history['train_recon_loss']) + 1)

    ax1.plot(epochs, history['train_recon_loss'], 'b-', label='Train Recon Loss')
    ax1.plot(epochs, history['train_pred_loss'], 'g-', label='Train Pred Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history['val_recon_psnr'], 'r--', label='Val Recon PSNR')
    ax2.plot(epochs, history['val_pred_psnr'], 'm--', label='Val Pred PSNR')
    ax2.set_ylabel('Validation PSNR (dB)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.suptitle(f'Training History - Fold {fold_idx}')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Training history plot saved to {save_path}")


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
        
        time_numeric = [i for i in range(len(time_points))]
        
        plt.subplot(2, 1, 1)
        plt.plot(time_numeric, pred_volumes, 'o-', label=f'Patient {patient_id}')
        plt.subplot(2, 1, 2) 
        plt.plot(time_numeric, gt_volumes, 's-', label=f'Patient {patient_id}')
    
    plt.subplot(2, 1, 1)
    plt.title('Predicted WMH Volume Progression')
    plt.xlabel('Time Point')
    plt.ylabel('WMH Volume (ml)')
    if volume_results:
        first_volumes = next(iter(volume_results.values()))
        time_points = first_volumes['time_points']
        plt.xticks(range(len(time_points)), time_points, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.title('Ground Truth WMH Volume Progression') 
    plt.xlabel('Time Point')
    plt.ylabel('WMH Volume (ml)')
    if volume_results:
        first_volumes = next(iter(volume_results.values()))
        time_points = first_volumes['time_points']
        plt.xticks(range(len(time_points)), time_points, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Volume progression plot saved to {save_path}")


# ============================================================
# === EVALUATION AND ANALYSIS FUNCTIONS ===
# ============================================================

def evaluate_and_visualize_tasks(model_path, source_loader, gt_loaders, device, original_scans_dir, results_dir="results"):
    """Evaluates a model on specific tasks and saves predictions as 3D NIfTI files."""
    print(f"\n--- Evaluating: {os.path.basename(model_path)} ---")

    model = ImageFlowNetODE(device=device, in_channels=2, ode_location='bottleneck', contrastive=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tasks = {
        "Interpolation_t1": {"scan_pair": "Scan2Wave3", "time": 1.0},
        "Prediction_t2":    {"scan_pair": "Scan3Wave4", "time": 2.0},
        "Extrapolation_t3": {"scan_pair": "Scan4Wave5", "time": 3.0},
    }
    metrics = {name: {
        "psnr": torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device),
        "ssim": torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    } for name in tasks}
    
    patient_predictions = {task_name: defaultdict(dict) for task_name in tasks}
    gt_iterators = {task: iter(loader) for task, loader in gt_loaders.items()}
    
    with torch.no_grad():
        for i, source_batch in enumerate(tqdm(source_loader, desc="Evaluating")):
            source_img = source_batch["source_image"].to(device)
            patient_ids, slice_indices = source_batch["patient_id"], source_batch["slice_idx"]

            for task_name, task_info in tasks.items():
                try:
                    t = torch.tensor([task_info["time"]], device=device)
                    pred_img = torch.sigmoid(model(source_img, t=t))

                    gt_pair_name = task_info["scan_pair"]
                    gt_batch = next(gt_iterators[gt_pair_name])
                    target_img = gt_batch["target_image"].to(device)
                    metrics[task_name]["psnr"].update(pred_img, target_img)
                    metrics[task_name]["ssim"].update(pred_img, target_img)
                    
                    for j in range(pred_img.shape[0]):
                        p_id = patient_ids[j]
                        s_idx = slice_indices[j].item()
                        pred_np = pred_img[j, 0].cpu().numpy()
                        patient_predictions[task_name][p_id][s_idx] = pred_np

                    if i == 0:
                        model_prefix = os.path.basename(model_path).split('.')[0]
                        visualize_results(
                            source=source_img, ground_truth=target_img, predicted=pred_img,
                            patient_ids=patient_ids, slice_indices=slice_indices,
                            filename=f"Comparison_{model_prefix}_to_{gt_pair_name}.png",
                            save_dir=results_dir
                        )
                except StopIteration:
                    continue

    print("\n💾 Saving 3D NIfTI volumes...")
    for task_name, predictions_by_patient in tqdm(patient_predictions.items(), desc="Saving"):
        gt_pair_name = tasks[task_name]["scan_pair"]
        model_prefix = os.path.basename(model_path).split('.')[0]
        save_dir = os.path.join(results_dir, f"{model_prefix}_Pred_{gt_pair_name}_3D")
        os.makedirs(save_dir, exist_ok=True)

        for patient_id, slices in predictions_by_patient.items():
            if not slices:
                continue

            max_slice_idx = max(slices.keys())
            H, W = next(iter(slices.values())).shape
            volume = np.zeros((H, W, max_slice_idx + 1), dtype=np.float32)

            for slice_idx, slice_data in slices.items():
                volume[:, :, slice_idx] = slice_data

            affine = np.eye(4)
            try:
                full_prefix = f"LBC36{patient_id}"
                original_file = next(f for f in os.listdir(original_scans_dir) if f.startswith(full_prefix))
                original_nii = nib.load(os.path.join(original_scans_dir, original_file))
                affine = original_nii.affine
            except:
                pass

            nii_image = nib.Nifti1Image(volume, affine)
            output_filename = os.path.join(save_dir, f"{patient_id}_predicted_3D.nii.gz")
            nib.save(nii_image, output_filename)

    final_results = {'model_path': model_path}
    for task_name in tasks:
        final_results[task_name] = {
            "PSNR": metrics[task_name]["psnr"].compute().item(),
            "SSIM": metrics[task_name]["ssim"].compute().item(),
        }
    return final_results


def load_folds_from_csv(fold_csv_path):
    """Load predefined patient folds from CSV."""
    df = pd.read_csv(fold_csv_path, dtype={"patient_ID": str})  
    folds = {}
    for _, row in df.iterrows():
        pid = str(row["patient_ID"])
        pid = re.sub(r"^LBC", "", pid)
        pid = re.sub(r"^LBC36", "", pid)
        pid = pid.zfill(4)                
        fold = int(row["fold"])
        folds.setdefault(fold, []).append(pid)
    return folds


# ============================================================
# === STAGE 2 - SEGMENTATION FUNCTIONS ===
# ============================================================

def train_segmentation(model, loader, opt, device):
    model.train()
    tot = 0
    for b in tqdm(loader, desc="[Stage2 Train]"):
        x, y = b["flair"].to(device), b["mask"].to(device)
        p = model(x)
        loss = dice_loss(p, y) + nn.functional.binary_cross_entropy(p, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot / len(loader)


def run_stage2_segmentation(pred_flair_dir, wmh_gt_dir, device, models_dir):
    """Run Stage 2 WMH segmentation."""
    from torch.utils.data import DataLoader
    
    ds = DownstreamSegmentationDataset(pred_flair_dir, wmh_gt_dir)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    
    model = SwinUNetSegmentation().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for ep in range(10):
        loss = train_segmentation(model, dl, opt, device)
        print(f"[Stage2] Epoch {ep+1}: loss={loss:.4f}")
    
    model_save_path = os.path.join(models_dir, "wmh_segmentation_swin_unet.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"[Stage2] Model saved to {model_save_path}")


def segment_3d_volume(model, volume_3d, device):
    """Segment a 3D volume slice by slice."""
    model.eval()
    segmented_volume = np.zeros_like(volume_3d)
    
    with torch.no_grad():
        for slice_idx in range(volume_3d.shape[2]):
            slice_data = volume_3d[:, :, slice_idx]
            if slice_data.max() - slice_data.min() > 1e-8:
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            
            slice_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_mask = model(slice_tensor)
            pred_mask_binary = (pred_mask > 0.5).float()
            segmented_volume[:, :, slice_idx] = pred_mask_binary.squeeze().cpu().numpy()
    
    return segmented_volume


def calculate_volume_ml(mask_volume, voxel_size_mm=(1.0, 1.0, 1.0)):
    """Calculate volume in milliliters from a binary mask volume."""
    voxel_volume_mm3 = voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]
    volume_mm3 = np.sum(mask_volume) * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0
    return volume_ml


def get_ground_truth_wmh_volume(wmh_dir, patient_id):
    """Load ground truth WMH volume for a specific patient."""
    matching_files = []
    for file in os.listdir(wmh_dir):
        if file.startswith(f"LBC36{patient_id}") and file.endswith('.nii.gz'):
            matching_files.append(file)
    
    if not matching_files:
        return None
    
    wmh_path = os.path.join(wmh_dir, matching_files[0])
    try:
        wmh_volume = nib.load(wmh_path).get_fdata(dtype=np.float32)
        return wmh_volume
    except Exception as e:
        print(f"⚠️ Could not load WMH for patient {patient_id}: {e}")
        return None


def analyze_wmh_volume_progression(predicted_flair_base_dir, gt_wmh_dirs, time_points, device):
    """Analyze WMH volume progression across time points."""
    volume_results = {}
    
    seg_model = SwinUNetSegmentation().to(device)
    seg_model_path = os.path.join(os.path.dirname(predicted_flair_base_dir), "wmh_segmentation_swin_unet.pth")
    
    if os.path.exists(seg_model_path):
        seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
        print("✅ Loaded trained segmentation model")
    else:
        print("⚠️ Segmentation model not found")
    
    seg_model.eval()
    
    pred_dirs_by_timepoint = {}
    for time_point in time_points:
        for folder in os.listdir(predicted_flair_base_dir):
            folder_path = os.path.join(predicted_flair_base_dir, folder)
            if os.path.isdir(folder_path) and f"Pred_{time_point}_3D" in folder:
                pred_dirs_by_timepoint[time_point] = folder_path
                break
    
    all_patients = set()
    for time_point, pred_dir in pred_dirs_by_timepoint.items():
        for pred_file in os.listdir(pred_dir):
            if pred_file.endswith('_predicted_3D.nii.gz'):
                match = re.match(r"(\d+)_predicted_3D\.nii\.gz", pred_file)
                if match:
                    all_patients.add(match.group(1))
    
    for patient_id in all_patients:
        print(f"📊 Analyzing patient {patient_id}")
        patient_volumes = {'predicted': [], 'ground_truth': [], 'time_points': []}
        
        for time_point in time_points:
            if time_point not in pred_dirs_by_timepoint:
                continue
            
            pred_dir = pred_dirs_by_timepoint[time_point]
            pred_file = f"{patient_id}_predicted_3D.nii.gz"
            pred_flair_path = os.path.join(pred_dir, pred_file)
            
            if not os.path.exists(pred_flair_path):
                continue
            
            try:
                pred_flair_volume = nib.load(pred_flair_path).get_fdata(dtype=np.float32)
                pred_wmh_volume = segment_3d_volume(seg_model, pred_flair_volume, device)
                pred_wmh_ml = calculate_volume_ml(pred_wmh_volume)
                
                gt_wmh_volume = get_ground_truth_wmh_volume(gt_wmh_dirs[time_point], patient_id)
                gt_wmh_ml = calculate_volume_ml(gt_wmh_volume) if gt_wmh_volume is not None else 0
                
                patient_volumes['predicted'].append(pred_wmh_ml)
                patient_volumes['ground_truth'].append(gt_wmh_ml)
                patient_volumes['time_points'].append(time_point)
                
            except Exception as e:
                print(f"⚠️ Error for patient {patient_id}: {e}")
                continue
        
        if patient_volumes['predicted']:
            volume_results[patient_id] = patient_volumes
    
    return volume_results
