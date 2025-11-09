# -*- coding: utf-8 -*-
"""
Experiment 3: FLAIR → FLAIR + WMH (Joint Training with Segmentation Loss)
"""

from base import BaseExperiment
from utils import (
    FLAIREvolutionDataset,
    LinearWarmupCosineAnnealingLR,
    neg_cos_sim,
    load_folds_from_csv,
    plot_fold_history,
    plot_volume_progression,
    calculate_volume_ml,
    get_ground_truth_wmh_volume,
    DiceLoss,
)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchmetrics
from collections import defaultdict
import nibabel as nib
from ImageFlowNet.src.nn.imageflownet_ode import ImageFlowNetODE


class ImageFlowNetODEWithSegmentation(nn.Module):
    """
    ImageFlowNet with an additional segmentation head for WMH prediction.
    
    Architecture:
        - Backbone: ImageFlowNetODE (predicts FLAIR)
        - Segmentation Head: Lightweight decoder (predicts WMH mask)
    """
    
    def __init__(self, device, in_channels=1, ode_location='bottleneck', contrastive=True):
        super().__init__()
        self.device = device
        
        # Backbone: ImageFlowNet for FLAIR prediction
        self.imageflownet = ImageFlowNetODE(
            device=device,
            in_channels=in_channels,
            ode_location=ode_location,
            contrastive=contrastive
        )
        
        # Segmentation Head: Simple decoder for WMH mask prediction
        # Takes the predicted FLAIR as input and outputs binary mask
        self.seg_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            # Output is logits, apply sigmoid during loss computation
        ).to(device)
    
    def forward(self, x, t, return_grad=False):
        """
        Forward pass through both FLAIR prediction and WMH segmentation.
        
        Args:
            x: Input FLAIR image [N, 1, H, W]
            t: Time delta
            return_grad: Whether to return gradient information
        
        Returns:
            pred_flair: Predicted FLAIR image [N, 1, H, W]
            pred_wmh_logits: Predicted WMH segmentation logits [N, 1, H, W]
        """
        # Predict FLAIR using ImageFlowNet
        if return_grad:
            pred_flair, vec_grad = self.imageflownet(x, t, return_grad=True)
            # Predict WMH mask from predicted FLAIR
            pred_wmh_logits = self.seg_head(pred_flair)
            return pred_flair, pred_wmh_logits, vec_grad
        else:
            pred_flair = self.imageflownet(x, t)
            # Predict WMH mask from predicted FLAIR
            pred_wmh_logits = self.seg_head(pred_flair)
            return pred_flair, pred_wmh_logits
    
    def freeze_time_independent(self):
        """Freeze time-independent parameters (for ODE training phase)."""
        self.imageflownet.freeze_time_independent()
    
    def time_independent_parameters(self):
        """Get time-independent parameters."""
        return self.imageflownet.time_independent_parameters()
    
    def simsiam_project(self, x):
        """Project to contrastive learning space."""
        return self.imageflownet.simsiam_project(x)
    
    def simsiam_predict(self, z):
        """Predict in contrastive learning space."""
        return self.imageflownet.simsiam_predict(z)


class Experiment3(BaseExperiment):
    """
    Experiment 3: FLAIR → FLAIR + WMH prediction with joint training.
    
    Key differences from Experiment 1:
    - Model outputs both FLAIR and WMH masks
    - Loss function includes both reconstruction loss and segmentation loss
    - Training is end-to-end, not two-stage
    """
    
    def __init__(self, experiment_number, experiment_config, config):
        """
        Initialize Experiment 3.
        
        Args:
            experiment_number: The experiment number
            experiment_config: Dictionary with experiment configuration
            config: Global configuration dictionary
        """
        super().__init__(experiment_number, experiment_config)
        self.config = config

        # Load test set patient IDs
        test_csv = self.config.get("TEST_CSV", None)
        if test_csv is None or not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test CSV not found at {test_csv}")
        
        self.test_patient_ids = self._load_patient_ids(test_csv, column="patient_ID")
        print(f"Loaded {len(self.test_patient_ids)} explicit test patients from {test_csv}")
    
    def _load_patient_ids(self, csv_path, column="patient_ID"):
        df = pd.read_csv(csv_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {csv_path}. Columns: {list(df.columns)}")
        return [str(x).strip() for x in df[column].astype(str).tolist()]
    
    def train_epoch_with_segmentation(self, model, loader, optimizer, ema, mse_loss, dice_loss, 
                                     device, epoch_idx, train_time_dependent, num_epochs, 
                                     contrastive_coeff, seg_loss_weight):
        """
        Training epoch for Experiment 3 with segmentation loss.
        
        Args:
            seg_loss_weight: Weight for segmentation loss (λ2 in the formula)
        """
        model.train()

        total_recon_loss = 0.0
        total_pred_loss = 0.0
        total_seg_loss = 0.0
        pred_loss_count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{num_epochs} [Train]")
        for i, batch in enumerate(pbar):
            source_flair = batch["source"][:, 0:1, :, :].to(device)  # Extract FLAIR channel only
            target_flair = batch["target"][:, 0:1, :, :].to(device)  # Extract FLAIR channel only
            target_wmh = batch["target"][:, 1:2, :, :].to(device) if batch["target"].shape[1] > 1 else None
            time_deltas = batch["time_delta"].to(device)

            optimizer.zero_grad()

            # ===== Phase 1: Reconstruction Loss =====
            # Unfreeze all parameters for reconstruction training
            for param in model.parameters():
                param.requires_grad = True

            source_recon_flair, source_recon_wmh = model(source_flair, t=torch.zeros(1).to(device))
            target_recon_flair, target_recon_wmh = model(target_flair, t=torch.zeros(1).to(device))
            
            # Apply sigmoid to FLAIR predictions for consistency
            source_recon_flair = torch.sigmoid(source_recon_flair)
            target_recon_flair = torch.sigmoid(target_recon_flair)
            
            # FLAIR reconstruction loss
            loss_recon_flair = mse_loss(source_recon_flair, source_flair) + mse_loss(target_recon_flair, target_flair)
            
            # Segmentation reconstruction loss (only if target WMH exists)
            loss_recon_seg = 0.0
            if target_wmh is not None:
                target_recon_wmh_prob = torch.sigmoid(target_recon_wmh)
                loss_recon_seg = dice_loss(target_recon_wmh_prob, target_wmh) + \
                                nn.functional.binary_cross_entropy(target_recon_wmh_prob, target_wmh)
            
            # Combined reconstruction loss
            loss_recon = loss_recon_flair + seg_loss_weight * loss_recon_seg

            # Contrastive loss (optional)
            if hasattr(model, 'simsiam_project') and hasattr(model, 'simsiam_predict'):
                z1 = model.simsiam_project(source_flair)
                z2 = model.simsiam_project(target_flair)
                p1 = model.simsiam_predict(z1)
                p2 = model.simsiam_predict(z2)
                loss_contrastive = neg_cos_sim(p1, z2)/2 + neg_cos_sim(p2, z1)/2
                loss = loss_recon + contrastive_coeff * loss_contrastive
            else:
                loss = loss_recon

            loss.backward()
            optimizer.step()
            ema.update()
            total_recon_loss += loss.item()
            total_seg_loss += loss_recon_seg if isinstance(loss_recon_seg, float) else loss_recon_seg.item()

            # ===== Phase 2: Prediction Loss (Time-Dependent) =====
            if train_time_dependent:
                optimizer.zero_grad()
                model.freeze_time_independent() if hasattr(model, 'freeze_time_independent') else None

                t = time_deltas[0:1]
                predicted_flair, predicted_wmh_logits = model(source_flair, t)
                
                # Apply sigmoid to FLAIR predictions for consistency
                predicted_flair = torch.sigmoid(predicted_flair)
                
                # FLAIR prediction loss
                loss_pred_flair = mse_loss(predicted_flair, target_flair)
                
                # WMH prediction loss
                loss_pred_seg = 0.0
                if target_wmh is not None:
                    predicted_wmh_prob = torch.sigmoid(predicted_wmh_logits)
                    loss_pred_seg = dice_loss(predicted_wmh_prob, target_wmh) + \
                                   nn.functional.binary_cross_entropy(predicted_wmh_prob, target_wmh)
                
                # Combined prediction loss
                loss_pred = loss_pred_flair + seg_loss_weight * loss_pred_seg

                loss_pred.backward()
                optimizer.step()
                ema.update()

                total_pred_loss += loss_pred.item()
                pred_loss_count += 1

            pbar.set_postfix(
                recon_loss=total_recon_loss / (i + 1),
                pred_loss=total_pred_loss / pred_loss_count if pred_loss_count > 0 else "N/A",
                seg_loss=total_seg_loss / (i + 1)
            )
        
        avg_recon_loss = total_recon_loss / len(loader)
        avg_pred_loss = total_pred_loss / pred_loss_count if pred_loss_count > 0 else float('nan')
        avg_seg_loss = total_seg_loss / len(loader)
        return avg_recon_loss, avg_pred_loss, avg_seg_loss
    
    def val_epoch_with_segmentation(self, model, loader, device):
        """
        Validation epoch for Experiment 3 with segmentation metrics.
        """
        model.eval()

        recon_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
        pred_psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)

        # Dice metric for segmentation - use BinaryDice with probabilities
        # from torchmetrics.classification import BinaryDice
        from utils import BinaryDice
        dice_metric = BinaryDice(threshold=0.5).to(device)
        saw_any_wmh = False

        with torch.no_grad():
            for batch in loader:
                source_flair = batch["source"][:, 0:1, :, :].to(device)
                target_flair = batch["target"][:, 0:1, :, :].to(device)
                target_wmh = batch["target"][:, 1:2, :, :].to(device) if batch["target"].shape[1] > 1 else None
                time_deltas = batch["time_delta"].to(device)

                # Reconstruction PSNR
                source_recon_flair, _ = model(source_flair, t=torch.zeros(1).to(device))
                target_recon_flair, _ = model(target_flair, t=torch.zeros(1).to(device))
                
                # Apply sigmoid for consistency with training
                source_recon_flair = torch.sigmoid(source_recon_flair)
                target_recon_flair = torch.sigmoid(target_recon_flair)
                
                recon_psnr_metric.update(source_recon_flair, source_flair)
                recon_psnr_metric.update(target_recon_flair, target_flair)

                # Prediction PSNR and Dice
                t = time_deltas[0:1]
                predicted_flair, predicted_wmh_logits = model(source_flair, t)
                
                # Apply sigmoid for consistency with training
                predicted_flair = torch.sigmoid(predicted_flair)
                
                pred_psnr_metric.update(predicted_flair, target_flair)
                
                # Segmentation Dice score - feed probabilities, not hard masks
                if target_wmh is not None:
                    saw_any_wmh = True
                    predicted_wmh_prob = torch.sigmoid(predicted_wmh_logits)
                    dice_metric.update(predicted_wmh_prob, target_wmh.int())

        avg_recon_psnr = recon_psnr_metric.compute().item()
        avg_pred_psnr = pred_psnr_metric.compute().item()
        avg_dice = dice_metric.compute().item() if saw_any_wmh else float('nan')

        return avg_recon_psnr, avg_pred_psnr, avg_dice
    
    def run(self):
        """Execute the full Experiment 3 pipeline."""
        print("\n" + "="*60)
        print("Starting Experiment 3: FLAIR → FLAIR + WMH (Joint Training)")
        print("="*60 + "\n")
        
        # Stage 1: Train models with joint FLAIR + WMH prediction
        self._train_imageflownet_with_segmentation()
        
        # Stage 2: Evaluate on test set
        self._evaluate_on_test_set()
        
        print(f"\n{'='*60}")
        print(f"🎉 Experiment {self.experiment_number} Complete!")
        print(f"{'='*60}")
    
    def _train_imageflownet_with_segmentation(self):
        """Train ImageFlowNet models with segmentation using cross-validation."""
        print("="*60)
        print("✅ Training ImageFlowNet with Joint FLAIR + WMH Prediction")
        print("="*60)
        
        # Define training pairs: t1 -> t3 only
        training_pairs = [
            ("Scan1Wave2", "Scan3Wave4", 2.0)  # Train on t1 -> t3
        ]
        
        # Initialize dataset - MUST have WMH masks as targets
        print("Initializing dataset with WMH targets for segmentation loss...")
        full_dataset = FLAIREvolutionDataset(
            root_dir=self.config["ROOT_DIR"],
            max_slices_per_patient=self.config["MAX_SLICES"],
            use_wmh=True,  # ✅ CRITICAL: Must be True to load WMH masks
            training_pairs=training_pairs
        )
        
        # Load fold assignments
        fold_csv = self.config["FOLD_CSV"]
        if not os.path.exists(fold_csv):
            raise FileNotFoundError(f"Fold CSV not found at {fold_csv}")
        folds_dict = load_folds_from_csv(fold_csv)
        print(f"Loaded patient folds from {fold_csv}")
        
        # K-Fold Cross-Validation Training
        print(f"\n📈 Starting {self.config['K_FOLDS']}-Fold Cross-Validation Training...")
        
        for val_fold_idx in self.config["CV_FOLDS"]:
            self._train_fold(val_fold_idx, full_dataset, folds_dict)
        
        print("\n✅ Training Complete!")
    
    def _train_fold(self, val_fold_idx, full_dataset, folds_dict):
        """Train a model on a single fold with segmentation."""
        print(f"\n{'='*50}")
        print(f"K-Fold Run: Validating on Fold {val_fold_idx}")
        print(f"{'='*50}\n")

        raw_val_pids = folds_dict[val_fold_idx]
        raw_train_pids = [pid for f_idx in self.config["CV_FOLDS"] if f_idx != val_fold_idx
                          for pid in folds_dict[f_idx]]
        
        test_set = set(self.test_patient_ids)
        val_pids = [pid for pid in raw_val_pids if pid not in test_set][:self.config["MAX_PATIENTS_PER_FOLD"]]
        train_pids = [pid for pid in raw_train_pids if pid not in test_set][
            : self.config["MAX_PATIENTS_PER_FOLD"] * (self.config["K_FOLDS"] - 1)
        ]
                
        print(f"Training patients:   {len(train_pids)}")
        print(f"Validation patients: {len(val_pids)}")
        
        # Data indices
        train_indices = [i for i, item in enumerate(full_dataset.index_map) 
                        if item['patient_id'] in set(train_pids)]
        val_indices = [i for i, item in enumerate(full_dataset.index_map) 
                      if item['patient_id'] in set(val_pids)]
        
        # DataLoaders
        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_indices),
            batch_size=self.config["BATCH_SIZE"],
            shuffle=False,
            num_workers=2
        )
        
        # Initialize model with segmentation head
        model = ImageFlowNetODEWithSegmentation(
            device=self.config["DEVICE"],
            in_channels=1,
            ode_location='bottleneck',
            contrastive=True
        ).to(self.config["DEVICE"])
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["LEARNING_RATE"])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config["NUM_EPOCHS"] // 10,
            max_epochs=self.config["NUM_EPOCHS"]
        )
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
        mse_loss = nn.MSELoss()
        dice_loss_fn = DiceLoss()
        
        best_val_psnr = 0.0
        recon_good_enough = False
        model_save_path = self.get_model_path(val_fold_idx)
        
        history = {
            'train_recon_loss': [],
            'train_pred_loss': [],
            'train_seg_loss': [],
            'train_recon_psnr': [],
            'train_pred_psnr': [],
            'val_recon_psnr': [],
            'val_pred_psnr': [],
            'val_dice': []
        }
        
        # Training loop
        for epoch in range(self.config["NUM_EPOCHS"]):
            avg_recon_loss, avg_pred_loss, avg_seg_loss = self.train_epoch_with_segmentation(
                model, train_loader, optimizer, ema, mse_loss, dice_loss_fn,
                self.config["DEVICE"], epoch, recon_good_enough,
                self.config["NUM_EPOCHS"], self.config["CONTRASTIVE_COEFF"],
                self.config.get("SEG_LOSS_WEIGHT", 1.0)  # Segmentation loss weight
            )
            
            with ema.average_parameters():
                val_recon_psnr, val_pred_psnr, val_dice = self.val_epoch_with_segmentation(
                    model, val_loader, self.config["DEVICE"]
                )
            
            with torch.no_grad(), ema.average_parameters():
                # Sample training set for faster evaluation
                train_sample_loader = DataLoader(
                    Subset(full_dataset, train_indices[:len(val_indices)]),
                    batch_size=self.config["BATCH_SIZE"],
                    shuffle=False
                )
                train_recon_psnr, train_pred_psnr, train_dice = self.val_epoch_with_segmentation(
                    model, train_sample_loader, self.config["DEVICE"]
                )

            history['train_recon_loss'].append(avg_recon_loss)
            history['train_pred_loss'].append(avg_pred_loss)
            history['train_seg_loss'].append(avg_seg_loss)
            history['train_recon_psnr'].append(train_recon_psnr)
            history['train_pred_psnr'].append(train_pred_psnr)
            history['val_recon_psnr'].append(val_recon_psnr)
            history['val_pred_psnr'].append(val_pred_psnr)
            history['val_dice'].append(val_dice)
            
            print(f"Epoch {epoch+1}: Train PSNR={train_pred_psnr:.4f}, Val PSNR={val_pred_psnr:.4f}, Val Dice={val_dice:.4f}")
            
            if not recon_good_enough and val_recon_psnr > self.config["RECON_PSNR_THR"]:
                recon_good_enough = True
                print("✅ Reconstruction threshold reached. Starting ODE training.")
            
            if val_pred_psnr > best_val_psnr:
                best_val_psnr = val_pred_psnr
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Val PSNR improved. Model saved to {model_save_path}")
            
            scheduler.step()
        
        # Plot training history using shared utility function
        # Convert history format to match utils.plot_fold_history expectations
        plot_fold_history(history, val_fold_idx, self.plots_dir)
        
        # Save training history to CSV
        history_df = pd.DataFrame({
            'epoch': range(1, len(history['train_recon_loss']) + 1),
            'train_recon_loss': history['train_recon_loss'],
            'train_pred_loss': history['train_pred_loss'],
            'train_seg_loss': history['train_seg_loss'],
            'train_recon_psnr': history['train_recon_psnr'],
            'train_pred_psnr': history['train_pred_psnr'],
            'val_recon_psnr': history['val_recon_psnr'],
            'val_pred_psnr': history['val_pred_psnr'],
            'val_dice': history['val_dice']
        })
        history_csv_path = os.path.join(self.results_dir, f"training_history_fold_{val_fold_idx}.csv")
        history_df.to_csv(history_csv_path, index=False)
        print(f"📊 Training history saved to {history_csv_path}")
    

    def _evaluate_on_test_set(self):
        """Evaluate trained models on test set and save predictions."""
        print("\n" + "="*60)
        print("✅ Evaluating on Held-Out Test Set")
        print("="*60)
        
        test_pids = self.test_patient_ids
        
        # Create datasets for evaluation (same as training)
        # Source: t1 (Scan1Wave2)
        source_dataset = FLAIREvolutionDataset(
            root_dir=self.config["ROOT_DIR"],
            max_slices_per_patient=self.config["MAX_SLICES"],
            use_wmh=True,  # Need WMH for evaluation
            training_pairs=[("Scan1Wave2", "Scan1Wave2", 0.0)]
        )
        
        # Targets: t2, t3, t4
        target_datasets = {}
        for target_scan in ["Scan2Wave3", "Scan3Wave4", "Scan4Wave5"]:
            target_datasets[target_scan] = FLAIREvolutionDataset(
                root_dir=self.config["ROOT_DIR"],
                max_slices_per_patient=self.config["MAX_SLICES"],
                use_wmh=True,
                training_pairs=[(target_scan, target_scan, 0.0)]
            )
        
        # Create dataloaders
        source_test_indices = [i for i, item in enumerate(source_dataset.index_map)
                               if item['patient_id'] in set(test_pids)]
        
        source_loader = DataLoader(
            Subset(source_dataset, source_test_indices),
            batch_size=self.config["BATCH_SIZE"],
            shuffle=False
        )
        
        gt_loaders = {}
        for scan_name, dataset in target_datasets.items():
            test_indices = [i for i, item in enumerate(dataset.index_map)
                           if item['patient_id'] in set(test_pids)]
            gt_loaders[scan_name] = DataLoader(
                Subset(dataset, test_indices),
                batch_size=self.config["BATCH_SIZE"],
                shuffle=False
            )
        
        # Get trained model paths
        model_paths = [
            self.get_model_path(i) for i in self.config["CV_FOLDS"]
            if os.path.exists(self.get_model_path(i))
        ]
        
        if not model_paths:
            print("⚠️ No trained models found to evaluate.")
            return
        
        # Evaluate each fold's model
        all_results = []
        for model_path in model_paths:
            results = self._evaluate_model_with_segmentation(
                model_path, source_loader, gt_loaders, target_datasets
            )
            all_results.append(results)
        
        # Aggregate and report results
        self._report_test_results(all_results)
        
        # Analyze WMH volume progression using predicted WMH masks
        print("\n" + "="*60)
        print("📊 Analyzing WMH Volume Progression")
        print("="*60)
        self._analyze_wmh_volumes_experiment3()
    
    def _analyze_wmh_volumes_experiment3(self):
        """
        Analyze WMH volume progression for Experiment 3.
        Unlike Experiment 1, this uses directly predicted WMH masks.
        """
        # Time points to analyze
        time_points = ["Scan2Wave3", "Scan3Wave4", "Scan4Wave5"]
        time_labels = {"Scan2Wave3": "t2", "Scan3Wave4": "t3", "Scan4Wave5": "t4"}
        
        # Get ground truth WMH directories
        gt_wmh_dirs = {}
        for scan in time_points:
            gt_wmh_dirs[time_labels[scan]] = os.path.join(
                self.config["ROOT_DIR"], f"{scan}_WMH_brain"
            )
        
        # Find predicted WMH directories (from the first model for simplicity)
        model_paths = [
            self.get_model_path(i) for i in self.config["CV_FOLDS"]
            if os.path.exists(self.get_model_path(i))
        ]
        
        if not model_paths:
            print("⚠️ No trained models found for volume analysis.")
            return
        
        # Use the first model's predictions for volume analysis
        model_prefix = os.path.basename(model_paths[0]).split('.')[0]
        
        # Collect volume data manually (since we have predicted WMH directly)
        volume_results = {}
        
        for patient_id in self.test_patient_ids:
            print(f"📊 Analyzing patient {patient_id}")
            patient_volumes = {'predicted': [], 'ground_truth': [], 'time_points': []}
            
            for scan in time_points:
                time_label = time_labels[scan]
                
                # Path to predicted WMH
                pred_wmh_dir = os.path.join(
                    self.results_dir, 
                    f"{model_prefix}_Pred_{scan}_WMH_3D"
                )
                pred_wmh_file = os.path.join(
                    pred_wmh_dir, 
                    f"{patient_id}_predicted_wmh_3D.nii.gz"
                )
                
                # Path to ground truth WMH
                gt_wmh_dir = gt_wmh_dirs[time_label]
                
                if not os.path.exists(pred_wmh_file):
                    print(f"  ⚠️ Predicted WMH not found for {time_label}")
                    continue
                
                try:
                    # Load predicted WMH volume with affine
                    pred_wmh_nii = nib.load(pred_wmh_file)
                    pred_wmh_volume = pred_wmh_nii.get_fdata(dtype=np.float32)
                    pred_affine = pred_wmh_nii.affine
                    
                    # Binarize predicted WMH (if not already binary)
                    pred_wmh_binary = (pred_wmh_volume > 0.5).astype(np.float32)
                    pred_wmh_ml = calculate_volume_ml(pred_wmh_binary, affine=pred_affine)
                    
                    # Load ground truth WMH volume with affine
                    gt_wmh_volume, gt_affine = get_ground_truth_wmh_volume(gt_wmh_dir, patient_id)
                    gt_wmh_ml = calculate_volume_ml(gt_wmh_volume, affine=gt_affine) if gt_wmh_volume is not None else 0
                    
                    patient_volumes['predicted'].append(pred_wmh_ml)
                    patient_volumes['ground_truth'].append(gt_wmh_ml)
                    patient_volumes['time_points'].append(time_label)
                    
                    print(f"  ✅ {time_label}: Pred={pred_wmh_ml:.2f}ml, GT={gt_wmh_ml:.2f}ml")
                    
                except Exception as e:
                    print(f"  ⚠️ Error for {time_label}: {e}")
                    continue
            
            if patient_volumes['predicted']:
                volume_results[patient_id] = patient_volumes
        
        # Plot volume progression
        if volume_results:
            plot_volume_progression(
                volume_results, 
                self.get_plots_path("wmh_volume_progression.png")
            )
            
            # Save results to CSV
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
            csv_path = self.get_results_path(f"wmh_volume_progression_{self.name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ Volume progression results saved to {csv_path}")
            
            # Print summary
            errors = [row['volume_error_ml'] for row in df_results]
            if errors:
                print(f"\n📊 Volume Analysis Summary:")
                print(f"   Mean Error: {np.mean(errors):.2f} ± {np.std(errors):.2f} ml")
                print(f"   Min Error: {np.min(errors):.2f} ml")
                print(f"   Max Error: {np.max(errors):.2f} ml")
        else:
            print("⚠️ No volume results to analyze")
    
    def _evaluate_model_with_segmentation(self, model_path, source_loader, gt_loaders, target_datasets):
        """Evaluate a single model on test set with segmentation."""
        print(f"\n--- Evaluating: {os.path.basename(model_path)} ---")
        
        # Load model
        model = ImageFlowNetODEWithSegmentation(
            device=self.config["DEVICE"],
            in_channels=1,
            ode_location='bottleneck',
            contrastive=True
        ).to(self.config["DEVICE"])
        model.load_state_dict(torch.load(model_path, map_location=self.config["DEVICE"]))
        model.eval()
        
        # Define evaluation tasks
        tasks = {
            "Interpolation_t2": {"scan_pair": "Scan2Wave3", "time": 1.0},
            "Training_t3": {"scan_pair": "Scan3Wave4", "time": 2.0},
            "Extrapolation_t4": {"scan_pair": "Scan4Wave5", "time": 3.0},
        }
        
        # Initialize metrics - use BinaryDice for consistency with validation
        # from torchmetrics.classification import BinaryDice
        from utils import BinaryDice
        metrics = {name: {
            "psnr_flair": torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(self.config["DEVICE"]),
            "ssim_flair": torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(self.config["DEVICE"]),
            "dice_wmh": BinaryDice(threshold=0.5).to(self.config["DEVICE"]),
        } for name in tasks}
        
        # Collect predictions for saving
        patient_predictions = {task_name: defaultdict(dict) for task_name in tasks}
        
        with torch.no_grad():
            for i, source_batch in enumerate(tqdm(source_loader, desc="Evaluating")):
                source_flair = source_batch["source"][:, 0:1, :, :].to(self.config["DEVICE"])
                patient_ids, slice_indices = source_batch["patient_id"], source_batch["slice_idx"]
                
                for task_name, task_info in tasks.items():
                    try:
                        # Predict
                        t = torch.tensor([task_info["time"]], device=self.config["DEVICE"])
                        pred_flair, pred_wmh_logits = model(source_flair, t=t)
                        pred_flair = torch.sigmoid(pred_flair)
                        pred_wmh_prob = torch.sigmoid(pred_wmh_logits)
                        
                        # Get ground truth - match by patient_id and slice_idx
                        gt_pair_name = task_info["scan_pair"]
                        gt_dataset = target_datasets[gt_pair_name]
                        
                        # Find matching ground truth slices for each sample in batch
                        target_flair_list = []
                        target_wmh_list = []
                        
                        for j in range(len(patient_ids)):
                            p_id = patient_ids[j]
                            s_idx = slice_indices[j].item()
                            
                            # Find the matching index in ground truth dataset
                            gt_idx = None
                            for idx, item in enumerate(gt_dataset.index_map):
                                if item['patient_id'] == p_id and item['slice_idx'] == s_idx:
                                    gt_idx = idx
                                    break
                            
                            if gt_idx is not None:
                                gt_sample = gt_dataset[gt_idx]
                                tgt = gt_sample["target"]  # (1, H, W) or (2, H, W)
                                
                                # Extract FLAIR channel
                                target_flair_list.append(tgt[0:1, :, :])
                                
                                # Extract WMH channel if it exists
                                if tgt.shape[0] > 1:
                                    target_wmh_list.append(tgt[1:2, :, :])
                                else:
                                    # WMH channel missing, use zeros
                                    target_wmh_list.append(torch.zeros_like(tgt[0:1, :, :]))
                            else:
                                # No matching ground truth found, use zeros
                                target_flair_list.append(torch.zeros_like(source_flair[j]))
                                target_wmh_list.append(torch.zeros_like(source_flair[j]))
                        
                        # Stack into tensors
                        target_flair = torch.stack(target_flair_list).to(self.config["DEVICE"])
                        target_wmh = torch.stack(target_wmh_list).to(self.config["DEVICE"])
                        
                        # Update metrics (BinaryDice uses probabilities with threshold=0.5)
                        metrics[task_name]["psnr_flair"].update(pred_flair, target_flair)
                        metrics[task_name]["ssim_flair"].update(pred_flair, target_flair)
                        metrics[task_name]["dice_wmh"].update(pred_wmh_prob, target_wmh.int())
                        
                        # Store predictions for 3D reconstruction
                        for j in range(pred_flair.shape[0]):
                            p_id = patient_ids[j]
                            s_idx = slice_indices[j].item()
                            patient_predictions[task_name][p_id][s_idx] = {
                                'flair': pred_flair[j, 0].cpu().numpy(),
                                'wmh': pred_wmh_prob[j, 0].cpu().numpy()
                            }
                        
                        # Visualize first batch
                        if i == 0:
                            self._visualize_predictions(
                                source_flair, target_flair, target_wmh,
                                pred_flair, pred_wmh_prob,
                                patient_ids, slice_indices,
                                model_path, task_name
                            )
                    except StopIteration:
                        continue
        
        # Compute final metrics
        final_results = {'model_path': model_path}
        for task_name in tasks:
            final_results[task_name] = {
                "PSNR": metrics[task_name]["psnr_flair"].compute().item(),
                "SSIM": metrics[task_name]["ssim_flair"].compute().item(),
                "Dice": metrics[task_name]["dice_wmh"].compute().item(),
            }
        
        # Save 3D predictions
        self._save_3d_predictions(patient_predictions, tasks, model_path)
        
        return final_results
    
    def _visualize_predictions(self, source_flair, target_flair, target_wmh,
                               pred_flair, pred_wmh, patient_ids, slice_indices,
                               model_path, task_name):
        """Visualize predictions for a batch."""
        import matplotlib.pyplot as plt
        
        model_prefix = os.path.basename(model_path).split('.')[0]
        save_path = os.path.join(self.results_dir, f"{model_prefix}_{task_name}_sample.png")
        
        n_samples = min(3, source_flair.shape[0])
        fig, axes = plt.subplots(n_samples, 5, figsize=(20, n_samples * 4))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            p_id = patient_ids[i]
            s_idx = slice_indices[i].item()
            
            # Column 1: Source FLAIR
            axes[i, 0].imshow(source_flair[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title(f"Source FLAIR\nPatient {p_id}, Slice {s_idx}")
            axes[i, 0].axis('off')
            
            # Column 2: Ground Truth FLAIR
            axes[i, 1].imshow(target_flair[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title("GT FLAIR")
            axes[i, 1].axis('off')
            
            # Column 3: Predicted FLAIR
            axes[i, 2].imshow(pred_flair[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 2].set_title("Predicted FLAIR")
            axes[i, 2].axis('off')
            
            # Column 4: Ground Truth WMH
            axes[i, 3].imshow(target_wmh[i, 0].cpu().numpy(), cmap='Reds', vmin=0, vmax=1)
            axes[i, 3].set_title("GT WMH")
            axes[i, 3].axis('off')
            
            # Column 5: Predicted WMH
            axes[i, 4].imshow(pred_wmh[i, 0].cpu().numpy(), cmap='Reds', vmin=0, vmax=1)
            axes[i, 4].set_title("Predicted WMH")
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📷 Sample visualization saved: {save_path}")
    
    def _save_3d_predictions(self, patient_predictions, tasks, model_path):
        """Save 3D FLAIR and WMH predictions as NIfTI files."""
        import nibabel as nib
        
        model_prefix = os.path.basename(model_path).split('.')[0]
        original_scans_dir = os.path.join(self.config["ROOT_DIR"], "Scan1Wave2_FLAIR_brain")
        
        for task_name, task_info in tasks.items():
            predictions_by_patient = patient_predictions[task_name]
            gt_pair_name = task_info["scan_pair"]
            
            # Create directories
            flair_save_dir = os.path.join(self.results_dir, f"{model_prefix}_Pred_{gt_pair_name}_FLAIR_3D")
            wmh_save_dir = os.path.join(self.results_dir, f"{model_prefix}_Pred_{gt_pair_name}_WMH_3D")
            os.makedirs(flair_save_dir, exist_ok=True)
            os.makedirs(wmh_save_dir, exist_ok=True)
            
            for patient_id, slices in predictions_by_patient.items():
                if not slices:
                    continue
                
                max_slice_idx = max(slices.keys())
                H, W = next(iter(slices.values()))['flair'].shape
                flair_volume = np.zeros((H, W, max_slice_idx + 1), dtype=np.float32)
                wmh_volume = np.zeros((H, W, max_slice_idx + 1), dtype=np.float32)
                
                for slice_idx, pred_data in slices.items():
                    flair_volume[:, :, slice_idx] = pred_data['flair']
                    wmh_volume[:, :, slice_idx] = pred_data['wmh']
                
                # Get affine from original scan
                affine = np.eye(4)
                try:
                    full_prefix = f"LBC36{patient_id}"
                    original_file = next(f for f in os.listdir(original_scans_dir) if f.startswith(full_prefix))
                    original_nii = nib.load(os.path.join(original_scans_dir, original_file))
                    affine = original_nii.affine
                except:
                    pass
                
                # Save FLAIR
                flair_nii = nib.Nifti1Image(flair_volume, affine)
                flair_path = os.path.join(flair_save_dir, f"{patient_id}_predicted_flair_3D.nii.gz")
                nib.save(flair_nii, flair_path)
                
                # Save WMH
                wmh_nii = nib.Nifti1Image(wmh_volume, affine)
                wmh_path = os.path.join(wmh_save_dir, f"{patient_id}_predicted_wmh_3D.nii.gz")
                nib.save(wmh_nii, wmh_path)
            
            print(f"💾 Saved 3D predictions for {task_name} in {flair_save_dir} and {wmh_save_dir}")
    
    def _report_test_results(self, all_results):
        """Aggregate and report test set results."""
        print("\n" + "="*60)
        print("📊 Final Test Set Results (Mean ± Std Dev)")
        print("="*60)
        
        # Extract metrics
        tasks = ["Interpolation_t2", "Training_t3", "Extrapolation_t4"]
        task_labels = {
            "Interpolation_t2": "Interpolation (t1→t2, Δt=1.0)",
            "Training_t3": "Training (t1→t3, Δt=2.0)",
            "Extrapolation_t4": "Extrapolation (t1→t4, Δt=3.0)"
        }
        
        for task in tasks:
            psnrs = [r[task]['PSNR'] for r in all_results]
            ssims = [r[task]['SSIM'] for r in all_results]
            dices = [r[task]['Dice'] for r in all_results]
            
            print(f"\n{task_labels[task]}:")
            print(f"  FLAIR PSNR: {np.mean(psnrs):.4f} ± {np.std(psnrs):.4f} dB")
            print(f"  FLAIR SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
            print(f"  WMH Dice:   {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        
        # Save to CSV
        test_results_data = []
        for i, result in enumerate(all_results):
            fold_idx = self.config["CV_FOLDS"][i]
            for task in tasks:
                test_results_data.append({
                    'fold': fold_idx,
                    'task': task,
                    'psnr_flair': result[task]['PSNR'],
                    'ssim_flair': result[task]['SSIM'],
                    'dice_wmh': result[task]['Dice']
                })
        
        test_results_df = pd.DataFrame(test_results_data)
        
        # Add summary statistics
        summary_rows = []
        for task in tasks:
            task_data = test_results_df[test_results_df['task'] == task]
            summary_rows.append({
                'fold': 'mean',
                'task': task,
                'psnr_flair': task_data['psnr_flair'].mean(),
                'ssim_flair': task_data['ssim_flair'].mean(),
                'dice_wmh': task_data['dice_wmh'].mean()
            })
            summary_rows.append({
                'fold': 'std',
                'task': task,
                'psnr_flair': task_data['psnr_flair'].std(),
                'ssim_flair': task_data['ssim_flair'].std(),
                'dice_wmh': task_data['dice_wmh'].std()
            })
        
        test_results_df = pd.concat([test_results_df, pd.DataFrame(summary_rows)], ignore_index=True)
        
        csv_path = os.path.join(self.results_dir, "test_set_evaluation_results.csv")
        test_results_df.to_csv(csv_path, index=False)
        print(f"\n📊 Test results saved to {csv_path}")
        print("="*60)


# ============================================================================
# === END OF EXPERIMENT 3 IMPLEMENTATION ===
# ============================================================================


# ============================================================
# === STANDALONE EXECUTION FOR EXPERIMENT 3 ===
# ============================================================

if __name__ == "__main__":
    """
    Run Experiment 3 directly without going through main.py
    Usage: python -m experiments.flair_to_flair_wmh
    """
    print("\n" + "="*70)
    print("🧪 Running Experiment 3: FLAIR → FLAIR + WMH (Standalone Mode)")
    print("="*70 + "\n")
    
    # Configuration
    CONFIG = {
        # Dataset
        # "ROOT_DIR": "/app/dataset/LBC1936",
        "ROOT_DIR": "/disk/febrian/Edinburgh_Data/LBC1936",
        "FOLD_CSV": "train_val_5fold.csv",
        "TEST_CSV": "test_set_patients.csv",
        
        # Training
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 1e-4,
        "NUM_EPOCHS": 100,
        "MAX_SLICES": 48,
        "MAX_PATIENTS_PER_FOLD": 5,
        
        # Thresholds and coefficients
        "RECON_PSNR_THR": 25.0,
        "CONTRASTIVE_COEFF": 0.1,
        "SEG_LOSS_WEIGHT": 1.0,  # ✅ NEW: Weight for segmentation loss (λ2)
        
        # Cross-validation
        "CV_FOLDS": [1, 2, 3, 4, 5],
        
        # Device
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    CONFIG["K_FOLDS"] = len(CONFIG["CV_FOLDS"])
    
    print(f"Using device: {CONFIG['DEVICE']}")
    
    # Experiment configuration
    experiment_config = {
        "name": "flair_to_flair_wmh",
        "description": "FLAIR → FLAIR + WMH with joint training",
        "use_wmh": True,  # ✅ Must be True for Experiment 3
        "class": Experiment3
    }
    
    # Run experiment
    experiment = Experiment3(
        experiment_number=3,
        experiment_config=experiment_config,
        config=CONFIG
    )
    experiment.run()
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT 3 COMPLETE")
    print("="*70 + "\n")