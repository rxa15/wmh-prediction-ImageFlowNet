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

from ImageFlowNet.src.nn.imageflownet_ode_flexible import ImageFlowNetODE_FlexibleOutput
from monai.networks.nets import SwinUNETR

import torch
from torch import nn

class BinaryDice(nn.Module):
    """
    Drop-in replacement for torchmetrics.classification.BinaryDice.
    Computes Dice coefficient for binary predictions.
    Supports `.update(preds, targets)` and `.compute()` for accumulated batches.
    
    Key improvements:
    - **Skips empty-empty cases** (both pred and GT are empty) from main Dice calculation
    - **Tracks false positives on empty slices** separately as FP rate
    - Accumulates intersection and cardinality across ALL batches (not per-batch averaging)
    - Properly handles probabilities with thresholding
    
    Usage:
        dice_metric = BinaryDice(threshold=0.5)
        dice_metric.update(preds, targets)
        dice_score = dice_metric.compute()  # Main Dice (skips empty-empty)
        fp_rate = dice_metric.compute_fp_rate()  # False positive rate on empty GT
        stats = dice_metric.get_stats()  # Detailed statistics
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate Dice components for one batch.
        Args:
            preds (torch.Tensor): Predicted probabilities (N, 1, H, W) - AFTER sigmoid
            targets (torch.Tensor): Binary ground truth masks (N, 1, H, W)
        """
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")

        # Threshold probabilities to get binary predictions
        preds_bin = (preds > self.threshold).float()
        targets_bin = (targets > 0.5).float()

        # Process each sample separately to track empty cases
        batch_size = preds.shape[0]
        for i in range(batch_size):
            pred_sample = preds_bin[i]
            target_sample = targets_bin[i]
            
            pred_has_fg = pred_sample.sum() > 0
            target_has_fg = target_sample.sum() > 0
            
            # Case 1: Both empty (skip from main Dice, track separately)
            if not pred_has_fg and not target_has_fg:
                self.empty_empty_count += 1
                continue
            
            # Case 2: GT empty but pred has WMH (false positive on empty slice)
            if pred_has_fg and not target_has_fg:
                self.empty_gt_with_pred_count += 1
                self.fp_pred_volume += pred_sample.sum().item()
            
            # Case 3: At least one has foreground - include in Dice calculation
            if pred_has_fg or target_has_fg:
                intersection = (pred_sample * target_sample).sum()
                pred_cardinality = pred_sample.sum()
                target_cardinality = target_sample.sum()
                
                self.intersection_sum += intersection.item()
                self.preds_card_sum += pred_cardinality.item()
                self.targets_card_sum += target_cardinality.item()
                self.valid_sample_count += 1

    def compute(self):
        """
        Return the mean Dice coefficient across non-empty samples.
        
        This is your PRIMARY metric for model selection.
        Empty-empty cases are excluded (as they don't provide useful signal).
        
        Returns 0.0 if no valid samples were found.
        """
        if self.valid_sample_count == 0:
            # No valid samples (only empty-empty cases or nothing at all)
            return torch.tensor(0.0, dtype=torch.float32)
        
        union = self.preds_card_sum + self.targets_card_sum
        
        if union == 0:
            # Should not happen if valid_sample_count > 0, but safety check
            return torch.tensor(0.0, dtype=torch.float32)
        
        dice = (2.0 * self.intersection_sum + self.eps) / (union + self.eps)
        return torch.tensor(dice, dtype=torch.float32)
    
    def compute_fp_rate(self):
        """
        Return the false positive rate on empty ground truth slices.
        
        This is your COMPANION metric to check for hallucinations.
        
        FP Rate = (# of empty GT slices where pred has WMH) / (# of empty GT slices)
        
        Returns 0.0 if there are no empty GT cases.
        """
        total_empty_gt = self.empty_empty_count + self.empty_gt_with_pred_count
        
        if total_empty_gt == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        
        fp_rate = self.empty_gt_with_pred_count / total_empty_gt
        return torch.tensor(fp_rate, dtype=torch.float32)
    
    def get_stats(self):
        """
        Return detailed statistics for analysis.
        
        Useful for debugging and understanding model behavior.
        """
        return {
            'valid_samples': self.valid_sample_count,  # Samples with at least one foreground
            'empty_empty': self.empty_empty_count,  # Both pred and GT empty (skipped)
            'false_positives': self.empty_gt_with_pred_count,  # GT empty but pred has WMH
            'fp_volume': self.fp_pred_volume,  # Total volume of false positive predictions
            'dice': self.compute().item(),
            'fp_rate': self.compute_fp_rate().item()
        }

    def reset(self):
        """Reset accumulated statistics."""
        # Dice computation (only non-empty cases)
        self.intersection_sum = 0.0
        self.preds_card_sum = 0.0
        self.targets_card_sum = 0.0
        self.valid_sample_count = 0  # Samples with at least one foreground
        
        # Empty case tracking
        self.empty_empty_count = 0  # Both pred and GT are empty
        self.empty_gt_with_pred_count = 0  # GT empty, but pred has WMH (false positive)
        self.fp_pred_volume = 0.0  # Total predicted volume on empty GT

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
    """
    Dice Loss for binary segmentation.
    
    IMPORTANT: Pass LOGITS (before sigmoid), not probabilities.
    This loss applies sigmoid internally.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (before sigmoid) - shape (N, 1, H, W)
            targets: Binary ground truth (0 or 1) - shape (N, 1, H, W)
        """
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
    
    Args:
        training_pairs: List of tuples [(source_scan, target_scan, time_delta)]
                       e.g., [("Scan1Wave2", "Scan3Wave4", 2.0)] for t1->t3 training
                       If None, loads all available pairs.
    """
    def __init__(self, root_dir, transform=None, max_slices_per_patient=None, use_wmh=True, training_pairs=None):
        self.root_dir = root_dir
        self.transform = transform
        self.use_wmh = use_wmh
        self.training_pairs = training_pairs
        self.index_map = []
        self.patient_ids = set()
        # Cache per-volume (min, max) to support volume-wise normalization
        self._vol_minmax_cache = {}

        print(f"📂 Scanning folders in {root_dir} ...")

        # Step 1: Identify FLAIR, WMH, and DEM folders by scan pair name
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
            elif "DEM" in folder:
                key = folder.split("_DEM")[0]
                folder_map[key]["DEM"] = folder_path

        print(f"✅ Found {len(folder_map)} scan-pair folders.")

        # Step 2: Build patient-level file mapping
        # Map: {patient_id: {scan_pair: {flair_path, wmh_path}}}
        patient_scans = defaultdict(lambda: defaultdict(dict))
        
        for scan_pair, paths in folder_map.items():
            if "FLAIR" not in paths:
                continue
            flair_folder = paths["FLAIR"]
            wmh_folder = paths.get("WMH", None)
            
            for f in os.listdir(flair_folder):
                if not f.endswith((".nii", ".nii.gz")):
                    continue
                fname = os.path.basename(f)
                match = re.match(r"LBC36(\d+)_(\d+)_", fname)
                if not match:
                    continue
                
                patient_id, _ = match.groups()
                flair_path = os.path.join(flair_folder, f)
                wmh_path = None
                if wmh_folder is not None:
                    # 1) Base prefix before "_FLAIR"
                    if "_FLAIR" in fname:
                        flair_base = fname.split("_FLAIR")[0]  # e.g., "LBC360002_1"
                    else:
                        # fallback: drop everything after the 2nd underscore
                        parts = fname.split("_")
                        if len(parts) >= 3:
                            flair_base = "_".join(parts[:2])    # "LBC360002_1"
                        else:
                            flair_base = fname.split(".nii")[0]

                    # 2) Look for any file in WMH folder that starts with the same base and contains "WMH"
                    for w in os.listdir(wmh_folder):
                        if not w.endswith((".nii", ".nii.gz")):
                            continue
                        if w.startswith(flair_base) and "WMH" in w:
                            wmh_path = os.path.join(wmh_folder, w)
                            break
                
                patient_scans[patient_id][scan_pair] = {
                    "flair_path": flair_path,
                    "wmh_path": wmh_path
                }

        # Step 3: Create training pairs
        if training_pairs is None:
            # Default: Load all consecutive pairs
            for patient_id, scans in patient_scans.items():
                for scan_pair in scans:
                    match = re.match(r"Scan(\d+)Wave(\d+)", scan_pair)
                    if not match:
                        continue
                    scan_idx, wave_idx = map(float, match.groups())
                    time_delta = wave_idx - scan_idx
                    
                    self._add_scan_pair(scans[scan_pair], scans[scan_pair], 
                                       patient_id, scan_pair, time_delta, max_slices_per_patient)
        else:
            # Custom training pairs (e.g., t1->t3)
            print(f"🎯 Using custom training pairs: {training_pairs}")
            for patient_id, scans in patient_scans.items():
                for source_scan, target_scan, time_delta in training_pairs:
                    if source_scan in scans and target_scan in scans:
                        self._add_scan_pair(scans[source_scan], scans[target_scan],
                                           patient_id, f"{source_scan}_to_{target_scan}", 
                                           time_delta, max_slices_per_patient)

        print(f"📊 Dataset ready. Found {len(self.index_map)} slices from {len(self.patient_ids)} patients.")
        print(f"🔧 Configuration: use_wmh = {self.use_wmh}")

    def _add_scan_pair(self, source_info, target_info, patient_id, scan_pair, time_delta, max_slices_per_patient):
        """Add a source->target scan pair to the dataset."""
        try:
            source_vol = nib.load(source_info["flair_path"]).get_fdata(dtype=np.float32)
            target_vol = nib.load(target_info["flair_path"]).get_fdata(dtype=np.float32)
            num_slices = min(source_vol.shape[2], target_vol.shape[2])
            
            slice_indices = list(range(14, num_slices))
            if max_slices_per_patient and len(slice_indices) > max_slices_per_patient:
                step = len(slice_indices) / max_slices_per_patient
                slice_indices = [int(i * step) for i in range(max_slices_per_patient)]

            for s_idx in slice_indices:
                self.index_map.append({
                    "patient_id": patient_id,
                    "scan_pair": scan_pair,
                    "source_flair_path": source_info["flair_path"],
                    "target_flair_path": target_info["flair_path"],
                    "source_wmh_path": source_info.get("wmh_path"),
                    "target_wmh_path": target_info.get("wmh_path"),
                    "slice_idx": s_idx,
                    "time_delta": time_delta
                })
                self.patient_ids.add(patient_id)
        except Exception as e:
            print(f"⚠️ Could not load scan pair: {e}")

    def _get_volume_minmax(self, file_path):
        """Return (vmin, vmax) for a 3D volume, cached per file path."""
        if file_path in self._vol_minmax_cache:
            return self._vol_minmax_cache[file_path]
        vol = nib.load(file_path).get_fdata(dtype=np.float32)
        vmin = float(np.min(vol))
        vmax = float(np.max(vol))
        self._vol_minmax_cache[file_path] = (vmin, vmax)
        return vmin, vmax

    def _load_slice(self, file_path, slice_idx, vmin=None, vmax=None):
        """Load a single slice with optional volume-wise normalization using (vmin, vmax)."""
        vol = nib.load(file_path).get_fdata(dtype=np.float32)
        img_slice = vol[:, :, slice_idx]
        if vmin is not None and vmax is not None:
            denom = (vmax - vmin)
            if denom > 1e-8:
                img_slice = (img_slice - vmin) / denom
        else:
            # Fallback: per-slice min-max (legacy behavior)
            smin = float(img_slice.min())
            smax = float(img_slice.max())
            if smax - smin > 1e-8:
                img_slice = (img_slice - smin) / (smax - smin)
        # return torch.from_numpy(img_slice).unsqueeze(0) # OLD
        # return torch.from_numpy(img_slice.copy()).unsqueeze(0) # ANOTHER OPTION
        return torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0) # RECOMMENDED

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        info = self.index_map[idx]
        slice_idx = info["slice_idx"]
        
        # Compute volume-wise min/max for source and target FLAIR
        s_vmin, s_vmax = self._get_volume_minmax(info["source_flair_path"])
        t_vmin, t_vmax = self._get_volume_minmax(info["target_flair_path"])

        # # Load source image (volume-wise normalized)
        # source_flair = self._load_slice(info["source_flair_path"], slice_idx, vmin=s_vmin, vmax=s_vmax)
        # if self.use_wmh and info["source_wmh_path"]:
        #     # For WMH masks, values are typically 0/1; keep as-is (no per-slice stretching)
        #     source_wmh = self._load_slice(info["source_wmh_path"], slice_idx, vmin=None, vmax=None)
        #     source_img = torch.cat([source_flair, source_wmh], dim=0)
        # else:
        #     source_img = source_flair
        
        # # Load target image (volume-wise normalized)
        # target_flair = self._load_slice(info["target_flair_path"], slice_idx, vmin=t_vmin, vmax=t_vmax)
        # if self.use_wmh and info["target_wmh_path"]:
        #     target_wmh = self._load_slice(info["target_wmh_path"], slice_idx, vmin=None, vmax=None)
        #     target_img = torch.cat([target_flair, target_wmh], dim=0)
        # else:
        #     target_img = target_flair

        # ---- Source ----
        source_flair = self._load_slice(info["source_flair_path"], slice_idx, vmin=s_vmin, vmax=s_vmax)
        if self.use_wmh:
            if info["source_wmh_path"]:
                source_wmh = self._load_slice(info["source_wmh_path"], slice_idx, vmin=None, vmax=None)
            else:
                # create an empty WMH channel if mask is missing
                source_wmh = torch.zeros_like(source_flair)
            source_img = torch.cat([source_flair, source_wmh], dim=0)   # always [2,H,W]
        else:
            source_img = source_flair                                   # always [1,H,W]

        # ---- Target ----
        target_flair = self._load_slice(info["target_flair_path"], slice_idx, vmin=t_vmin, vmax=t_vmax)
        if self.use_wmh:
            if info["target_wmh_path"]:
                target_wmh = self._load_slice(info["target_wmh_path"], slice_idx, vmin=None, vmax=None)
            else:
                target_wmh = torch.zeros_like(target_flair)
            target_img = torch.cat([target_flair, target_wmh], dim=0)   # always [2,H,W]
        else:
            target_img = target_flair                                   # always [1,H,W]

        # after building source_img and target_img...
        source_img = source_img.contiguous().clone()
        target_img = target_img.contiguous().clone()
        
        return {
            "source": source_img,
            "target": target_img,
            "time_delta": torch.tensor(info["time_delta"], dtype=torch.float32),
            "patient_id": info["patient_id"],
            "slice_idx": torch.tensor(info["slice_idx"], dtype=torch.long)
        }


class DownstreamSegmentationDataset(Dataset):
    """Loads 2D slices from corresponding 3D predicted FLAIR and 3D ground truth WMH volumes."""
    def __init__(self, pred_flair_dir_3d, wmh_gt_dir):
        self.index_map = []
        # Cache for volume-wise min/max per file path
        self._vol_minmax_cache = {}
        
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
        # Use cached volume min/max to normalize slices consistently
        if file_path not in self._vol_minmax_cache:
            vol = nib.load(file_path).get_fdata(dtype=np.float32)
            vmin = float(np.min(vol))
            vmax = float(np.max(vol))
            self._vol_minmax_cache[file_path] = (vmin, vmax, vol)
        else:
            vmin, vmax, vol = self._vol_minmax_cache[file_path]

        img_slice = vol[:, :, slice_idx]
        denom = (vmax - vmin)
        if denom > 1e-8:
            img_slice = (img_slice - vmin) / denom
        # return torch.from_numpy(img_slice).unsqueeze(0) # OLD
        # return torch.from_numpy(img_slice.copy()).unsqueeze(0) # ANOTHER OPTION
        return torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0) # RECOMMENDED

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

def train_epoch(
    model,
    loader,
    optimizer,
    ema,
    recon_loss,
    device,
    epoch_idx,
    train_time_dependent,
    num_epochs,
    contrastive_coeff,
    wmh_lambda=1.0,   # ✅ new: weight for WMH-focused loss (can set to 0 to disable)
):
    model.train()

    total_recon_loss = 0.0
    total_pred_loss = 0.0
    pred_loss_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{num_epochs} [Train]")
    for i, batch in enumerate(pbar):
        # -------------------------------
        # 1) Get inputs and split channels
        # -------------------------------
        source_all = batch["source"].to(device)  # [B, C, H, W]
        target_all = batch["target"].to(device)  # [B, C, H, W]
        time_deltas = batch["time_delta"].to(device)

        # Assume channel 0 = FLAIR, channel 1 = WMH (if present)
        source_flair = source_all[:, 0:1, ...]
        target_flair = target_all[:, 0:1, ...]

        if source_all.shape[1] > 1:
            # Use WMH at target time point as supervision for future lesion
            wmh_mask = (target_all[:, 1:2, ...] > 0).float()  # [B,1,H,W]
            # Per-sample flag: does this slice have any WMH?
            has_wmh = (wmh_mask.view(wmh_mask.size(0), -1).sum(dim=1) > 0)
        else:
            wmh_mask = None
            has_wmh = None

        # Ground-truth delta only on FLAIR
        gt_delta_flair = target_flair - source_flair

        optimizer.zero_grad()

        # -------------------------------
        # 2) Reconstruction Loss (t = 0)
        # -------------------------------
        if hasattr(model, 'unfreeze'):
            model.unfreeze()

        zeros_t = torch.zeros(1, device=device)
        # Model sees both FLAIR & WMH (2 ch), but predicts only FLAIR reconstruction (1 ch)
        source_recon = model(source_all, t=zeros_t) # input: [B, 2, H, W]
        target_recon = model(target_all, t=zeros_t) # input: [B, 2, H, W]

        # Global reconstruction loss over full FLAIR image
        loss_src_global = recon_loss(source_recon, source_flair)
        loss_tgt_global = recon_loss(target_recon, target_flair)
        loss_recon_global = loss_src_global + loss_tgt_global

        # WMH-focused reconstruction loss (only where WMH > 0)
        if wmh_mask is not None and has_wmh.any():
            # pick only samples that actually have WMH
            mask_nonempty = wmh_mask[has_wmh]             # [B_w,1,H,W]
            src_rec_wmh   = source_recon[has_wmh] * mask_nonempty
            src_gt_wmh    = source_flair[has_wmh] * mask_nonempty
            tgt_rec_wmh   = target_recon[has_wmh] * mask_nonempty
            tgt_gt_wmh    = target_flair[has_wmh] * mask_nonempty

            loss_src_wmh = recon_loss(src_rec_wmh, src_gt_wmh)
            loss_tgt_wmh = recon_loss(tgt_rec_wmh, tgt_gt_wmh)
            loss_recon_wmh = loss_src_wmh + loss_tgt_wmh
        else:
            loss_recon_wmh = torch.tensor(0.0, device=device)

        # Combine global + WMH loss
        loss_recon = loss_recon_global + wmh_lambda * loss_recon_wmh

        # -------------------------------
        # 3) Optional SimSiam contrastive loss
        # -------------------------------
        if hasattr(model, 'simsiam_project') and hasattr(model, 'simsiam_predict'):
            # Use only FLAIR for representation learning
            z1 = model.simsiam_project(source_all)
            z2 = model.simsiam_project(target_all)
            p1 = model.simsiam_predict(z1)
            p2 = model.simsiam_predict(z2)
            loss_contrastive = neg_cos_sim(p1, z2) / 2 + neg_cos_sim(p2, z1) / 2
            loss = loss_recon + contrastive_coeff * loss_contrastive
        else:
            loss_contrastive = torch.tensor(0.0, device=device)
            loss = loss_recon

        # -------------------------------
        # 4) Backprop reconstruction step
        # -------------------------------
        loss.backward()
        optimizer.step()
        ema.update()
        total_recon_loss += loss.item()

        # -------------------------------
        # 5) Time-dependent Prediction Loss (ODE)
        # -------------------------------
        if train_time_dependent:
            optimizer.zero_grad()
            if hasattr(model, 'freeze_time_independent'):
                model.freeze_time_independent()

            # Use the same time delta for the whole batch (as in your original code)
            t = time_deltas[0:1]

            # Predict future FLAIR image
            predicted_target_flair = model(source_all, t) # input: [B, 2, H, W]
            predicted_delta_flair  = predicted_target_flair - source_flair

            # Global prediction loss over delta
            loss_pred_global = recon_loss(predicted_delta_flair, gt_delta_flair)

            # WMH-focused prediction loss (delta only inside WMH)
            if wmh_mask is not None and has_wmh is not None and has_wmh.any():
                mask_nonempty = wmh_mask[has_wmh]
                pred_delta_wmh = predicted_delta_flair[has_wmh] * mask_nonempty
                gt_delta_wmh   = gt_delta_flair[has_wmh] * mask_nonempty
                loss_pred_wmh  = recon_loss(pred_delta_wmh, gt_delta_wmh)
            else:
                loss_pred_wmh = torch.tensor(0.0, device=device)

            loss_pred = loss_pred_global + wmh_lambda * loss_pred_wmh

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
    avg_pred_loss  = total_pred_loss / pred_loss_count if pred_loss_count > 0 else float('nan')
    return avg_recon_loss, avg_pred_loss


def val_epoch(model, loader, device):
    model.eval()

    recon_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    pred_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    delta_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)

    with torch.no_grad():
        for batch in loader:
            source_all = batch["source"].to(device)  # [B, C, H, W]
            target_all = batch["target"].to(device)  # [B, C, H, W]
            time_deltas = batch["time_delta"].to(device)

            # Use only FLAIR (channel 0) for evaluation
            source_flair = source_all[:, 0:1, ...]
            target_flair = target_all[:, 0:1, ...]

            gt_delta_flair = target_flair - source_flair

            # Reconstruction PSNR (t = 0)
            zeros_t = torch.zeros(1, device=device)
            source_recon = model(source_all, t=zeros_t) # input: [B, 2, H, W]
            target_recon = model(target_all, t=zeros_t) # input: [B, 2, H, W]
            recon_psnr_metric.update(source_recon, source_flair)
            recon_psnr_metric.update(target_recon, target_flair)

            # Prediction PSNR (t = Δt)
            t = time_deltas[0:1]  # same convention as in train_epoch
            predicted_target_flair = model(source_all, t)
            predicted_delta_flair = predicted_target_flair - source_flair

            pred_psnr_metric.update(predicted_target_flair, target_flair)
            delta_psnr_metric.update(predicted_delta_flair, gt_delta_flair)

    avg_recon_psnr = recon_psnr_metric.compute().item()
    avg_pred_psnr = pred_psnr_metric.compute().item()
    avg_delta_psnr = delta_psnr_metric.compute().item()

    return avg_recon_psnr, avg_pred_psnr, avg_delta_psnr



# ============================================================
# === VISUALIZATION FUNCTIONS ===
# ============================================================

def visualize_results(source, ground_truth, predicted, patient_ids, slice_indices, n_samples=5, filename="test_results.png", save_dir="results", overlay_type=None):
    """
    Visualize comparison between source, ground truth, and predictions.
    
    Args:
        source: Source images tensor (N, C, H, W)
        ground_truth: Ground truth images tensor (N, C, H, W)
        predicted: Predicted images tensor (N, C, H, W)
        patient_ids: List of patient IDs
        slice_indices: Slice indices being visualized
        n_samples: Number of samples to visualize (default 5)
        filename: Output filename
        save_dir: Directory to save results
        overlay_type: Type of overlay for 2nd channel - 'wmh', 'dem', or None (default: None)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # Set overlay colormap and label based on type
    if overlay_type == 'wmh':
        overlay_cmap = 'Reds'
        overlay_label = 'WMH'
    elif overlay_type == 'dem':
        overlay_cmap = 'Blues'
        overlay_label = 'DEM'
    else:
        overlay_cmap = 'Reds'  # Default to red
        overlay_label = 'Overlay'

    plt.figure(figsize=(15, n_samples * 5))
    n_samples = min(n_samples, source.shape[0])

    for i in range(n_samples):
        p_id = patient_ids[i]
        s_idx = slice_indices[i].item()

        # Column 1: Source image(s)
        plt.subplot(n_samples, 3, i * 3 + 1)
        source_flair = source[i, 0].cpu().numpy()
        plt.imshow(source_flair, cmap='gray')
        
        # Overlay second channel if exists (WMH or DEM)
        if source.shape[1] > 1:
            source_overlay = source[i, 1].cpu().numpy()
            plt.imshow(source_overlay, cmap=overlay_cmap, alpha=0.4)
            plt.title(f"Patient: {p_id}\nSource FLAIR + {overlay_label} (Slice: {s_idx})")
        else:
            plt.title(f"Patient: {p_id}\nSource FLAIR (Slice: {s_idx})")
        plt.axis('off')

        # Column 2: Ground Truth
        plt.subplot(n_samples, 3, i * 3 + 2)
        gt_flair = ground_truth[i, 0].cpu().numpy()
        plt.imshow(gt_flair, cmap='gray')
        
        # Overlay second channel if exists
        if ground_truth.shape[1] > 1:
            gt_overlay = ground_truth[i, 1].cpu().numpy()
            plt.imshow(gt_overlay, cmap=overlay_cmap, alpha=0.4)
            plt.title(f"Ground Truth FLAIR + {overlay_label}")
        else:
            plt.title("Ground Truth FLAIR")
        plt.axis('off')

        # Column 3: Prediction
        plt.subplot(n_samples, 3, i * 3 + 3)
        pred_flair = predicted[i, 0].cpu().numpy()
        plt.imshow(pred_flair, cmap='gray')
        
        # Overlay second channel if exists
        if predicted.shape[1] > 1:
            pred_overlay = predicted[i, 1].cpu().numpy()
            plt.imshow(pred_overlay, cmap=overlay_cmap, alpha=0.4)
            plt.title(f"Prediction FLAIR + {overlay_label}")
        else:
            plt.title("Prediction FLAIR")
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
    
    # Check if history is empty
    if not history['train_recon_loss']:
        print(f"⚠️ Skipping plot for fold {fold_idx}: No history data.")
        plt.close()
        return
        
    epochs = range(1, len(history['train_recon_loss']) + 1)

    # Axis 1: Losses
    ax1.plot(epochs, history['train_recon_loss'], 'b-', label='Train Recon Loss')
    # Note: 'train_pred_loss' is now the "Delta Loss"
    ax1.plot(epochs, history['train_pred_loss'], 'g-', label='Train Delta Loss') 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(bottom=0)

    # Axis 2: PSNRs
    ax2 = ax1.twinx()
    
    # Plot Recon and Pred PSNRs (Diagnostic Metrics)
    ax2.plot(epochs, history['val_recon_psnr'], 'r--', label='Val Recon PSNR')
    ax2.plot(epochs, history['val_pred_psnr'], 'm--', label='Val Pred PSNR')
    
    if 'train_recon_psnr' in history:
        ax2.plot(epochs, history['train_recon_psnr'], 'r:', alpha=0.6, label='Train Recon PSNR')
    if 'train_pred_psnr' in history:
        ax2.plot(epochs, history['train_pred_psnr'], 'm:', alpha=0.6, label='Train Pred PSNR')
    
    if 'val_delta_psnr' in history:
        ax2.plot(epochs, history['val_delta_psnr'], 'c--', linewidth=2, label='**Val Delta PSNR**')
        
    if 'train_delta_psnr' in history:
        ax2.plot(epochs, history['train_delta_psnr'], 'c:', alpha=0.7, label='Train Delta PSNR')

    ax2.set_ylabel('PSNR (dB)', color='r') # Changed from "Validation PSNR"
    ax2.tick_params(axis='y', labelcolor='r')

    # Legend and Layout
    fig.suptitle(f'Training History - Fold {fold_idx}')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Training history plot saved to {save_path}")
    
    # Save history data to CSV
    history_data = {'epoch': list(epochs)}
    for key, values in history.items():
        history_data[key] = values
    
    history_df = pd.DataFrame(history_data)
    csv_path = os.path.join(save_dir, f"training_history_fold_{fold_idx}.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"📊 Training history data saved to {csv_path}")


def plot_volume_progression(volume_results, save_path="volume_progression.png"):
    """Plot WMH volume progression for all patients and save data to CSV"""
    import pandas as pd
    
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
    
    # Save volume progression data to CSV
    volume_data = []
    for patient_id, volumes in volume_results.items():
        for i, time_point in enumerate(volumes['time_points']):
            volume_data.append({
                'patient_id': patient_id,
                'time_point': time_point,
                'predicted_wmh_ml': volumes['predicted'][i],
                'ground_truth_wmh_ml': volumes['ground_truth'][i],
                'volume_error_ml': volumes['predicted'][i] - volumes['ground_truth'][i],
                'absolute_error_ml': abs(volumes['predicted'][i] - volumes['ground_truth'][i])
            })
    
    volume_df = pd.DataFrame(volume_data)
    csv_path = save_path.replace('.png', '.csv')
    volume_df.to_csv(csv_path, index=False)
    print(f"📊 Volume progression data saved to {csv_path}")


# ============================================================
# === EVALUATION AND ANALYSIS FUNCTIONS ===
# ============================================================

def evaluate_and_visualize_tasks(model_path, source_loader, gt_loaders, device, original_scans_dir, results_dir="results"):
    """Evaluates a model on specific tasks and saves predictions as 3D NIfTI files."""
    print(f"\n--- Evaluating: {os.path.basename(model_path)} ---")

    model = ImageFlowNetODE_FlexibleOutput(device=device, in_channels=2, out_channels=1, ode_location='bottleneck', contrastive=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Updated tasks for t1->t2 (interpolation), t1->t3 (training), t1->t4 (extrapolation)
    tasks = {
        "Interpolation_t2": {"scan_pair": "Scan2Wave3", "time": 1.0},  # t1 -> t2 (Δt=1.0)
        "Training_t3":      {"scan_pair": "Scan3Wave4", "time": 2.0},  # t1 -> t3 (Δt=2.0, trained)
        "Extrapolation_t4": {"scan_pair": "Scan4Wave5", "time": 3.0},  # t1 -> t4 (Δt=3.0)
    }
    metrics = {name: {
        "psnr": torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device),
        "ssim": torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    } for name in tasks}
    
    patient_predictions = {task_name: defaultdict(dict) for task_name in tasks}
    gt_iterators = {task: iter(loader) for task, loader in gt_loaders.items()}
    
    with torch.no_grad():
        for i, source_batch in enumerate(tqdm(source_loader, desc="Evaluating")):
            source_all = source_batch["source"].to(device)   # [B, C, H, W]
            patient_ids, slice_indices = source_batch["patient_id"], source_batch["slice_idx"]

            for task_name, task_info in tasks.items():
                try:
                    t = torch.tensor([task_info["time"]], device=device)
                    pred_img = model(source_all, t=t)  # model sees FLAIR + WMH

                    gt_pair_name = task_info["scan_pair"]
                    gt_batch = next(gt_iterators[gt_pair_name])
                    target_all = gt_batch["target"].to(device)
                    target_flair = target_all[:, 0:1, ...]

                    # PSNR/SSIM on FLAIR only
                    metrics[task_name]["psnr"].update(pred_img, target_flair)
                    metrics[task_name]["ssim"].update(pred_img, target_flair)

                    # Store predictions per patient and slice
                    for b_idx in range(pred_img.size(0)):
                        pid = patient_ids[b_idx]
                        s_idx = int(slice_indices[b_idx].item())
                        patient_predictions[task_name][pid][s_idx] = (
                            pred_img[b_idx, 0].detach().cpu().numpy()
                        )

                    # For visualization, you can still pass the *full* tensors
                    if i == 0:
                        model_prefix = os.path.basename(model_path).split('.')[0]
                        visualize_results(
                            source=source_all,          # keeps WMH overlay if present
                            ground_truth=target_all,
                            predicted=pred_img,         # 1-channel FLAIR
                            patient_ids=patient_ids,
                            slice_indices=slice_indices,
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


def run_stage2_segmentation(pred_flair_dir, wmh_gt_dir, device, models_dir, epoch_num=10):
    """Run Stage 2 WMH segmentation."""
    from torch.utils.data import DataLoader
    
    ds = DownstreamSegmentationDataset(pred_flair_dir, wmh_gt_dir)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    
    model = SwinUNetSegmentation().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for ep in range(epoch_num):
        loss = train_segmentation(model, dl, opt, device)
        print(f"[Stage2] Epoch {ep+1}: loss={loss:.4f}")
    
    model_save_path = os.path.join(models_dir, "wmh_segmentation_swin_unet.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"[Stage2] Model saved to {model_save_path}")


def segment_3d_volume(model, volume_3d, device):
    """Segment a 3D volume slice by slice."""
    model.eval()
    segmented_volume = np.zeros_like(volume_3d)
    # Compute volume-wise min/max once for consistent normalization
    vmin = float(np.min(volume_3d))
    vmax = float(np.max(volume_3d))
    denom = vmax - vmin
    
    with torch.no_grad():
        for slice_idx in range(volume_3d.shape[2]):
            slice_data = volume_3d[:, :, slice_idx]
            if denom > 1e-8:
                slice_data = (slice_data - vmin) / denom
            
            # slice_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0).float().to(device)
            slice_tensor = torch.from_numpy(slice_data.copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_mask = model(slice_tensor)
            pred_mask_binary = (pred_mask > 0.5).float()
            segmented_volume[:, :, slice_idx] = pred_mask_binary.squeeze().cpu().numpy()
    
    return segmented_volume


def calculate_volume_ml(mask_volume, affine=None, voxel_size_mm=None):
    """
    Calculate volume in milliliters from a binary mask volume.
    
    Args:
        mask_volume: Binary numpy array (3D mask)
        affine: 4x4 affine matrix from NIfTI header (preferred method)
        voxel_size_mm: Tuple (sx, sy, sz) - manual override if you already know spacing
    
    Returns:
        Volume in milliliters (ml)
    
    Note:
        If affine is provided, voxel sizes are extracted as the norms of affine column vectors.
        Otherwise falls back to voxel_size_mm, or (1.0, 1.0, 1.0) as last resort.
    """
    if voxel_size_mm is None and affine is not None:
        # Extract voxel sizes from affine matrix (norms of column vectors)
        sx = np.linalg.norm(affine[:3, 0])
        sy = np.linalg.norm(affine[:3, 1])
        sz = np.linalg.norm(affine[:3, 2])
        voxel_size_mm = (sx, sy, sz)
    
    if voxel_size_mm is None:
        # Fallback (not recommended for real data)
        voxel_size_mm = (1.0, 1.0, 1.0)
    
    voxel_volume_mm3 = voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]
    volume_mm3 = np.sum(mask_volume) * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0
    return volume_ml


def get_ground_truth_wmh_volume(wmh_dir, patient_id):
    """
    Load ground truth WMH volume for a specific patient.
    
    Returns:
        Tuple of (wmh_volume, affine) or (None, None) if not found
    """
    matching_files = []
    for file in os.listdir(wmh_dir):
        if file.startswith(f"LBC36{patient_id}") and file.endswith('.nii.gz'):
            matching_files.append(file)
    
    if not matching_files:
        return None, None
    
    wmh_path = os.path.join(wmh_dir, matching_files[0])
    try:
        wmh_nii = nib.load(wmh_path)
        wmh_volume = wmh_nii.get_fdata(dtype=np.float32)
        affine = wmh_nii.affine
        return wmh_volume, affine
    except Exception as e:
        print(f"⚠️ Could not load WMH for patient {patient_id}: {e}")
        return None, None


def analyze_wmh_volume_progression(predicted_flair_base_dir, model_dir, gt_wmh_dirs, time_points, device):
    """Analyze WMH volume progression across time points."""
    volume_results = {}
    
    seg_model = SwinUNetSegmentation().to(device)
    # seg_model_path = os.path.join(os.path.dirname(predicted_flair_base_dir), "wmh_segmentation_swin_unet.pth")
    seg_model_path = os.path.join(model_dir, "wmh_segmentation_swin_unet.pth")
    
    print("model_dir:", model_dir)
    print("seg_model_path:", seg_model_path)
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
                # Load predicted FLAIR with affine
                pred_flair_nii = nib.load(pred_flair_path)
                pred_flair_volume = pred_flair_nii.get_fdata(dtype=np.float32)
                pred_affine = pred_flair_nii.affine
                
                # Segment the predicted FLAIR to get WMH
                pred_wmh_volume = segment_3d_volume(seg_model, pred_flair_volume, device)
                pred_wmh_ml = calculate_volume_ml(pred_wmh_volume, affine=pred_affine)
                
                # Load ground truth WMH with affine
                gt_wmh_volume, gt_affine = get_ground_truth_wmh_volume(gt_wmh_dirs[time_point], patient_id)
                gt_wmh_ml = calculate_volume_ml(gt_wmh_volume, affine=gt_affine) if gt_wmh_volume is not None else 0
                
                patient_volumes['predicted'].append(pred_wmh_ml)
                patient_volumes['ground_truth'].append(gt_wmh_ml)
                patient_volumes['time_points'].append(time_point)
                
            except Exception as e:
                print(f"⚠️ Error for patient {patient_id}: {e}")
                continue
        
        if patient_volumes['predicted']:
            volume_results[patient_id] = patient_volumes
    
    return volume_results


# ============================================================
# === STAGE 2 - INFERENCE ONLY USING PRETRAINED SWINUNETR ===
# ============================================================

def run_stage2_inference_only(pred_flair_dir, wmh_gt_dir, pretrained_model_path, time_point_label, results_dir):
    """
    Run Stage 2 WMH segmentation using PRETRAINED SwinUNETR (inference only, no training).
    
    Args:
        pred_flair_dir: Directory containing predicted 3D FLAIR images
        wmh_gt_dir: Directory containing ground truth WMH masks
        pretrained_model_path: Path to pretrained SwinUNETR model (.pth file)
        time_point_label: Label for this time point (e.g., "Scan3Wave4")
        results_dir: Directory to save results
    
    Returns:
        Dictionary containing evaluation metrics and volume results
    """
    from torch.utils.data import DataLoader
    from monai.metrics import DiceMetric
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (224, 224)
    
    print(f"\n{'='*60}")
    print("🔬 Stage 2: WMH Segmentation Inference (Pretrained SwinUNETR)")
    print(f"{'='*60}")
    print(f"📂 Predicted FLAIR: {pred_flair_dir}")
    print(f"📂 Ground truth WMH: {wmh_gt_dir}")
    print(f"🤖 Model: {pretrained_model_path}")
    print(f"⏰ Time point: {time_point_label}")
    
    # 1. Load pretrained SwinUNETR model (matching training setup exactly)
    try:
        from monai.networks.nets import SwinUNETR
        from monai.transforms import Compose, Resized, Lambdad, EnsureTyped
        
        # NOTE: Must match training setup exactly (no img_size parameter)
        model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            spatial_dims=2,
            feature_size=48,
            use_checkpoint=True
        ).to(device)
        
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        model.eval()
        print("✅ Loaded pretrained SwinUNETR model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"💡 Make sure the model architecture matches your training setup!")
        return None
    
    # 2. Define transforms for resizing (matching training setup)
    val_transforms = Compose([
        Resized(keys=["flair", "mask"], spatial_size=IMG_SIZE, mode=("bilinear", "nearest")),
        Lambdad(keys="mask", func=lambda x: (x > 0.5).float()),
        EnsureTyped(keys=["flair", "mask"], dtype=torch.float)
    ])
    
    # 3. Create dataset with transforms
    class DownstreamSegmentationDatasetWithTransforms(Dataset):
        """Dataset for Stage 2 with MONAI transforms for resizing."""
        def __init__(self, pred_flair_dir_3d, wmh_gt_dir, transforms=None):
            self.index_map = []
            self.transforms = transforms
            # Cache volumes and volume-wise min/max per path
            self._vol_cache = {}
            
            print(f"📂 Loading Stage 2 dataset...")
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
                        pred_nii = nib.load(pred_path)
                        gt_nii = nib.load(gt_path)
                        num_slices = pred_nii.shape[2]
                        voxel_dims = pred_nii.header.get_zooms()[:3]
                        voxel_volume_mm3 = np.prod(voxel_dims)
                        
                        for s_idx in range(num_slices):
                            self.index_map.append({
                                "patient_id": patient_id_num,
                                "pred_path": pred_path,
                                "gt_path": gt_path,
                                "slice_idx": s_idx,
                                "voxel_volume_mm3": voxel_volume_mm3
                            })
                    except Exception as e:
                        print(f"⚠️ Could not process {pred_file}: {e}")
            
            print(f"✅ Loaded {len(self.index_map)} slices for Stage 2 inference")
        
        def __len__(self):
            return len(self.index_map)
        
        def _load_slice(self, file_path, slice_idx):
            # Load volume once and normalize slice using volume-wise min-max
            if file_path not in self._vol_cache:
                vol = nib.load(file_path).get_fdata(dtype=np.float32)
                vmin = float(np.min(vol))
                vmax = float(np.max(vol))
                self._vol_cache[file_path] = (vol, vmin, vmax)
            else:
                vol, vmin, vmax = self._vol_cache[file_path]

            img_slice = vol[:, :, slice_idx]
            denom = (vmax - vmin)
            if denom > 1e-8:
                img_slice = (img_slice - vmin) / denom
            return img_slice[None, :, :].copy()  # Add channel dimension
        
        def __getitem__(self, idx):
            info = self.index_map[idx]
            flair_slice = self._load_slice(info["pred_path"], info["slice_idx"])
            mask_slice = self._load_slice(info["gt_path"], info["slice_idx"])
            
            data = {"flair": flair_slice, "mask": mask_slice}
            
            if self.transforms:
                data = self.transforms(data)
            
            return {
                "flair": torch.as_tensor(data["flair"], dtype=torch.float),
                "mask": torch.as_tensor(data["mask"], dtype=torch.float),
                "patient_id": info["patient_id"],
                "voxel_volume_mm3": torch.tensor(info["voxel_volume_mm3"], dtype=torch.float32)
            }
    
    # 4. Create dataset and dataloader
    try:
        ds = DownstreamSegmentationDatasetWithTransforms(pred_flair_dir, wmh_gt_dir, transforms=val_transforms)
        if len(ds) == 0:
            print("❌ Dataset is empty")
            return None
        dl = DataLoader(ds, batch_size=4, shuffle=False)
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return None
    
    # 5. Run inference and collect results
    print("\n🔍 Running inference...")
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    all_slice_results = []
    
    with torch.no_grad():
        for batch in tqdm(dl, desc="[Stage 2 Inference]"):
            x = batch["flair"].to(device)
            y = batch["mask"].to(device)
            patient_ids = batch["patient_id"]
            voxel_volumes_mm3 = batch["voxel_volume_mm3"].to(device)
            
            # Forward pass
            p_logits = model(x)
            p_probs = torch.sigmoid(p_logits)
            p_binarized = (p_probs > 0.5).float()
            
            # Update Dice metric
            dice_metric(y_pred=p_binarized, y=y)
            
            # Compute WMH volumes
            def compute_wmh_volume_ml(mask, voxel_volumes):
                mask_bin = (mask > 0.5).float()
                voxel_count = mask_bin.sum(dim=[1, 2, 3])
                volume_mm3 = voxel_count * voxel_volumes
                volume_ml = volume_mm3 / 1000.0
                return volume_ml.cpu().numpy()
            
            pred_volumes_ml = compute_wmh_volume_ml(p_probs, voxel_volumes_mm3)
            gt_volumes_ml = compute_wmh_volume_ml(y, voxel_volumes_mm3)
            
            for j in range(len(patient_ids)):
                all_slice_results.append({
                    "patient_id": patient_ids[j],
                    "predicted_volume_ml": pred_volumes_ml[j],
                    "ground_truth_volume_ml": gt_volumes_ml[j],
                })
    
    # 6. Compute final metrics
    test_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    print(f"\n{'='*60}")
    print(f"🎯 Stage 2 Results - {time_point_label}")
    print(f"{'='*60}")
    print(f"Dice Score: {test_dice:.4f}")
    
    # 7. Save per-slice results
    if all_slice_results:
        slice_df = pd.DataFrame(all_slice_results)
        slice_df['volume_error_ml'] = slice_df['predicted_volume_ml'] - slice_df['ground_truth_volume_ml']
        slice_csv_path = os.path.join(results_dir, f'stage2_volume_per_slice_{time_point_label}.csv')
        slice_df.to_csv(slice_csv_path, index=False)
        print(f"📊 Per-slice results: {slice_csv_path}")
        
        # 8. Aggregate per-patient results
        patient_df = slice_df.groupby('patient_id').sum().reset_index()
        patient_df['time_point'] = time_point_label
        patient_df = patient_df[[
            'patient_id', 'time_point', 'predicted_volume_ml',
            'ground_truth_volume_ml', 'volume_error_ml'
        ]]
        
        patient_csv_path = os.path.join(results_dir, f'stage2_volume_per_patient_{time_point_label}.csv')
        patient_df.to_csv(patient_csv_path, index=False)
        print(f"📊 Per-patient results: {patient_csv_path}")
        
        # 9. Print summary statistics
        errors = patient_df['volume_error_ml']
        print(f"\n📊 Volume Statistics (Per-Patient):")
        print(f"   Mean Error: {errors.mean():.2f} ± {errors.std():.2f} ml")
        print(f"   Mean GT Volume: {patient_df['ground_truth_volume_ml'].mean():.2f} ml")
        print(f"   Mean Pred Volume: {patient_df['predicted_volume_ml'].mean():.2f} ml")
    
    # 10. Save visualization sample
    try:
        model.eval()
        sample_batch = next(iter(dl))
        x_sample = sample_batch["flair"].to(device)
        y_sample = sample_batch["mask"].to(device)
        
        with torch.no_grad():
            p_sample = torch.sigmoid(model(x_sample))
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(x_sample[0].cpu().squeeze().numpy(), cmap='gray')
        plt.title("Predicted FLAIR")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(y_sample[0].cpu().squeeze().numpy(), cmap='jet')
        plt.title("Ground Truth WMH")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow((p_sample[0].cpu().squeeze().numpy() > 0.5).astype(float), cmap='jet')
        plt.title("Predicted WMH")
        plt.axis('off')
        
        plt.tight_layout()
        sample_path = os.path.join(results_dir, f"stage2_sample_{time_point_label}.png")
        plt.savefig(sample_path)
        plt.close()
        print(f"🖼️  Sample visualization: {sample_path}")
    except Exception as e:
        print(f"⚠️ Could not save visualization: {e}")
    
    print(f"{'='*60}\n")
    
    return {
        'dice_score': test_dice,
        'patient_results': patient_df if all_slice_results else None,
        'time_point': time_point_label
    }