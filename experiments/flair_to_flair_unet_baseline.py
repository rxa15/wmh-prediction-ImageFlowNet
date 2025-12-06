# -*- coding: utf-8 -*-
"""
Experiment 9: FLAIR → FLAIR with UNet baseline and downstream WMH segmentation.
"""

from base import BaseExperiment
from utils_laras import (
    FLAIREvolutionDataset,
    LinearWarmupCosineAnnealingLR,
    load_folds_from_csv,
    visualize_results,
    plot_fold_history,
    run_stage2_segmentation,
    analyze_wmh_volume_progression,
    plot_volume_progression,
)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchmetrics
from collections import defaultdict
import nibabel as nib
torch.cuda.empty_cache()
import pandas as pd
import numpy as np
from tqdm import tqdm
from unet import UNet


# ============================================================
# === CUSTOM TRAINING/VALIDATION FOR UNET BASELINE ===
# ============================================================

def train_epoch_unet(model, loader, optimizer, ema, recon_loss, device, epoch_idx, num_epochs):
    """
    Train UNet for one epoch using MSE loss.
    UNet is a simple image-to-image model without time dependency.
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{num_epochs} [Train]")
    for batch in pbar:
        source_all = batch["source"].to(device)
        target_all = batch["target"].to(device)
        
        # Extract FLAIR channel only
        source_flair = source_all[:, 0:1, ...]
        target_flair = target_all[:, 0:1, ...]
        
        optimizer.zero_grad()
        
        # UNet directly predicts target from source (no time parameter)
        predicted_target = model(source_flair)
        
        # MSE loss between predicted and ground truth target
        loss = recon_loss(predicted_target, target_flair)
        
        loss.backward()
        optimizer.step()
        ema.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(loader)
    return avg_loss


def val_epoch_unet(model, loader, device):
    """
    Validate UNet model using PSNR metric.
    """
    model.eval()
    
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    with torch.no_grad():
        for batch in loader:
            source_all = batch["source"].to(device)
            target_all = batch["target"].to(device)
            
            # Extract FLAIR channel only
            source_flair = source_all[:, 0:1, ...]
            target_flair = target_all[:, 0:1, ...]
            
            # UNet prediction
            predicted_target = model(source_flair)
            
            # Update PSNR metric
            psnr_metric.update(predicted_target, target_flair)
    
    avg_psnr = psnr_metric.compute().item()
    return avg_psnr


def evaluate_and_visualize_tasks_unet(model_path, source_loader, gt_loaders, device, original_scans_dir, results_dir="results"):
    """
    Evaluate UNet model on test tasks and save predictions as 3D NIfTI files.
    """
    print(f"\n--- Evaluating: {os.path.basename(model_path)} ---")
    
    model = UNet(in_channels=1, out_channels=1, bilinear=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Tasks for evaluation
    tasks = {
        "Interpolation_t2": {"scan_pair": "Scan2Wave3", "time": 1.0},
        "Training_t3":      {"scan_pair": "Scan3Wave4", "time": 2.0},
        "Extrapolation_t4": {"scan_pair": "Scan4Wave5", "time": 3.0},
    }
    
    metrics = {name: {
        "psnr": torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device),
        "ssim": torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    } for name in tasks}
    
    patient_predictions = {task_name: defaultdict(dict) for task_name in tasks}
    gt_iterators = {task: iter(loader) for task, loader in gt_loaders.items()}
    
    with torch.no_grad():
        for i, source_batch in enumerate(tqdm(source_loader, desc="Evaluating")):
            source_all = source_batch["source"].to(device)
            patient_ids, slice_indices = source_batch["patient_id"], source_batch["slice_idx"]
            
            # Use only FLAIR for the model
            source_flair = source_all[:, 0:1, ...]
            
            for task_name, task_info in tasks.items():
                try:
                    # UNet directly predicts target (no time parameter)
                    pred_img = model(source_flair)
                    
                    gt_pair_name = task_info["scan_pair"]
                    gt_batch = next(gt_iterators[gt_pair_name])
                    target_all = gt_batch["target"].to(device)
                    target_flair = target_all[:, 0:1, ...]
                    
                    # Update metrics
                    metrics[task_name]["psnr"].update(pred_img, target_flair)
                    metrics[task_name]["ssim"].update(pred_img, target_flair)
                    
                    # Store predictions
                    for b_idx in range(pred_img.size(0)):
                        pid = patient_ids[b_idx]
                        s_idx = int(slice_indices[b_idx].item())
                        patient_predictions[task_name][pid][s_idx] = (
                            pred_img[b_idx, 0].detach().cpu().numpy()
                        )
                    
                    # Visualization
                    if i == 0:
                        model_prefix = os.path.basename(model_path).split('.')[0]
                        visualize_results(
                            source=source_flair,
                            ground_truth=target_flair,
                            predicted=pred_img,
                            patient_ids=patient_ids,
                            slice_indices=slice_indices,
                            filename=f"Comparison_{model_prefix}_to_{gt_pair_name}.png",
                            save_dir=results_dir
                        )
                except StopIteration:
                    continue
    
    # Save 3D NIfTI volumes
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
            except Exception as e:
                print(f"Warning: Could not load original NIfTI for {patient_id}: {e}")
            
            nii_img = nib.Nifti1Image(volume, affine)
            save_path = os.path.join(save_dir, f"LBC36{patient_id}.nii.gz")
            nib.save(nii_img, save_path)
    
    # Compute final metrics
    results = {
        'model_path': model_path,
    }
    for task_name in tasks:
        psnr_val = metrics[task_name]["psnr"].compute().item()
        ssim_val = metrics[task_name]["ssim"].compute().item()
        results[task_name] = {"PSNR": psnr_val, "SSIM": ssim_val}
        print(f"{task_name}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")
    
    return results


class Experiment9(BaseExperiment):
    """
    Experiment 1: FLAIR → FLAIR prediction with downstream WMH segmentation.
    Uses only L1 loss for segmentation
    """
    
    def __init__(self, experiment_number, experiment_config, config):
        """
        Initialize Experiment 1.
        
        Args:
            experiment_number: The experiment number
            experiment_config: Dictionary with experiment configuration
            config: Global configuration dictionary
        """
        super().__init__(experiment_number, experiment_config)
        self.config = config

        # print("self.config['TEST_CSV']:", self.config["TEST_CSV"])
        # print("self.config['TEST_CSV', None]:", self.config["TEST_CSV", None])

        # Load test set patient IDs
        test_csv = self.config["TEST_CSV"]
        if test_csv is None or not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test CSV not found at {test_csv}")
        
        self.test_patient_ids = self._load_patient_ids(test_csv, column="patient_ID")
        print(f"Loaded {len(self.test_patient_ids)} explicit test patients from {test_csv}")
    
    def _load_patient_ids(self, csv_path, column="patient_ID"):
        df = pd.read_csv(csv_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {csv_path}. Columns: {list(df.columns)}")
        return [str(x).strip() for x in df[column].astype(str).tolist()]
    
    # def _diagnose_wmh(self, dataset):
    #     print("\n================ WMH DIAGNOSTIC ================")

    #     num_total = len(dataset)
    #     num_with_wmh = 0
    #     patients_with_wmh = set()

    #     for i in range(num_total):
    #         item = dataset[i]
    #         target = item["target"]

    #         # If dataset loads only FLAIR (1 channel), skip
    #         if target.shape[0] == 1:
    #             continue

    #         wmh_mask = target[1]  # Channel 1 = WMH
    #         if wmh_mask.sum() > 0:
    #             num_with_wmh += 1
    #             patients_with_wmh.add(item["patient_id"])

    #     print(f"Total slices              : {num_total}")
    #     print(f"Slices with WMH > 0       : {num_with_wmh}")
    #     print(f"Patients with WMH slices  : {len(patients_with_wmh)}")
    #     print(f"List of patients          : {sorted(list(patients_with_wmh))}")

    #     if num_with_wmh == 0:
    #         print("⚠️  NO WMH SLICES FOUND!")
    #         print("Possible reasons:")
    #         print(" - use_wmh=False somewhere")
    #         print(" - WMH paths missing in dataset")
    #         print(" - Slices with WMH got skipped due to slice selection (14:)")
    #         print(" - Wrong scan pairs (t1->t3 has no WMH available)")
    #         print("=================================================\n")
    #     else:
    #         print("✅ WMH found in dataset.")
    #         print("=================================================\n")
    
    def run(self):
        """Execute the full Experiment 1 pipeline."""
        print("\n" + "="*60)
        print("Starting Experiment 1: FLAIR → FLAIR (L1 only)")
        print("="*60 + "\n")
        
        # Stage 1: Train ImageFlowNet models
        predicted_flair_dir, ground_truth_wmh_dir = self._stage1_train_imageflownet()
        
        # Stage 2: WMH Segmentation from predicted FLAIR
        if predicted_flair_dir and ground_truth_wmh_dir:
            self._stage2_wmh_segmentation(predicted_flair_dir, ground_truth_wmh_dir)
        else:
            print("[Stage 2] Skipped because Stage 1 did not complete successfully.")
    
    def _stage1_train_imageflownet(self):
        """Train ImageFlowNet models using cross-validation."""
        print("="*60)
        print("✅ STAGE 1: ImageFlowNet Training")
        print("="*60)
        
        # Define training pairs: t1 -> t3 only
        # Scan1Wave2 (t1) -> Scan3Wave4 (t3) with time_delta = 2.0
        training_pairs = [
            ("Scan1Wave2", "Scan3Wave4", 2.0)  # Train on t1 -> t3
        ]
        
        # Initialize dataset with custom training pairs
        print("Initializing dataset with custom training pairs (t1->t3)...")
        full_dataset = FLAIREvolutionDataset(
            root_dir=self.config["ROOT_DIR"],
            max_slices_per_patient=self.config["MAX_SLICES"],
            use_wmh=self.use_wmh,
            training_pairs=training_pairs  # Only train on t1->t3
        )

        # self._diagnose_wmh(full_dataset)
        
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
        
        # Final Evaluation
        return self._evaluate_on_test_set(full_dataset, folds_dict)
    
    def _train_fold(self, val_fold_idx, full_dataset, folds_dict):
        """Train a model on a single fold."""
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
            num_workers=0
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_indices),
            batch_size=self.config["BATCH_SIZE"],
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        model = UNet(
            in_channels=1,
            out_channels=1,
            bilinear=True
        ).to(self.config["DEVICE"])
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["LEARNING_RATE"])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config["NUM_EPOCHS"] // 10,
            max_epochs=self.config["NUM_EPOCHS"]
        )
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
        recon_loss = nn.MSELoss()
        
        best_val_psnr = 0.0
        model_save_path = self.get_model_path(val_fold_idx)
        
        history = {
            'train_loss': [],
            'train_psnr': [],
            'val_psnr': []
        }
        
        # Training loop - will run for NUM_EPOCHS (e.g., 50 epochs)
        for epoch in range(self.config["NUM_EPOCHS"]):
            # Train for this epoch
            avg_loss = train_epoch_unet(
                model, train_loader, optimizer, ema, recon_loss,
                self.config["DEVICE"], epoch, self.config["NUM_EPOCHS"]
            )
            
            # Validation with EMA parameters
            with ema.average_parameters():
                val_psnr = val_epoch_unet(model, val_loader, self.config["DEVICE"])
            
            # Training PSNR (sample for efficiency)
            with torch.no_grad(), ema.average_parameters():
                train_sample_loader = DataLoader(
                    Subset(full_dataset, train_indices[:len(val_indices)]),
                    batch_size=self.config["BATCH_SIZE"],
                    shuffle=False
                )
                train_psnr = val_epoch_unet(model, train_sample_loader, self.config["DEVICE"])
            
            # Record history
            history['train_loss'].append(avg_loss)
            history['train_psnr'].append(train_psnr)
            history['val_psnr'].append(val_psnr)
            
            print(f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}: "
                  f"Train Loss={avg_loss:.4f}, Train PSNR={train_psnr:.4f}, Val PSNR={val_psnr:.4f}")
            
            # Save best model
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Val PSNR improved to {val_psnr:.4f}. Model saved to {model_save_path}")
            
            scheduler.step()
        
        # Plot training history
        plot_fold_history(history, val_fold_idx, self.plots_dir)
        
        # Save training history to CSV
        history_df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'train_psnr': history['train_psnr'],
            'val_psnr': history['val_psnr'],
        })
        history_csv_path = os.path.join(self.results_dir, f"training_history_fold_{val_fold_idx}.csv")
        history_df.to_csv(history_csv_path, index=False)
        print(f"📊 Training history saved to {history_csv_path}")
    
    def _evaluate_on_test_set(self, full_dataset, folds_dict):
        """Evaluate all trained models on the held-out test set."""
        print("\n" + "="*60)
        print("✅ CV Training Complete. Starting Final Evaluation on Held-Out Test Set.")
        print("="*60)
        
        test_pids = self.test_patient_ids
        
        # Create separate datasets for source and each target timepoint
        # Source: t1 (Scan1Wave2)
        source_dataset = FLAIREvolutionDataset(
            root_dir=self.config["ROOT_DIR"],
            max_slices_per_patient=self.config["MAX_SLICES"],
            use_wmh=self.use_wmh,
            training_pairs=[("Scan1Wave2", "Scan1Wave2", 0.0)]  # Source only
        )
        
        # Targets: t2, t3, t4 (Scan2Wave3, Scan3Wave4, Scan4Wave5)
        target_datasets = {}
        for target_scan in ["Scan2Wave3", "Scan3Wave4", "Scan4Wave5"]:
            target_datasets[target_scan] = FLAIREvolutionDataset(
                root_dir=self.config["ROOT_DIR"],
                max_slices_per_patient=self.config["MAX_SLICES"],
                use_wmh=self.use_wmh,
                training_pairs=[(target_scan, target_scan, 0.0)]  # Target only
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
            print("No trained models found to evaluate.")
            return None, None
        
        original_scans_dir = os.path.join(self.config["ROOT_DIR"], "Scan1Wave2_FLAIR_brain")

        all_results = [
            evaluate_and_visualize_tasks_unet(
                path, source_loader, gt_loaders,
                self.config["DEVICE"],
                original_scans_dir=original_scans_dir,
                results_dir=self.results_dir
            )
            for path in model_paths
        ]
        
        # Report results
        print("\n" + "="*60)
        print("============= Final Test Set Results (Mean +/- Std Dev) =============")
        print("="*60)
        
        interp_psnrs = [r['Interpolation_t2']['PSNR'] for r in all_results]
        train_psnrs = [r['Training_t3']['PSNR'] for r in all_results]
        extrap_psnrs = [r['Extrapolation_t4']['PSNR'] for r in all_results]
        
        print(f"Interpolation PSNR (t1->t2, Δt=1.0): {np.mean(interp_psnrs):.4f} +/- {np.std(interp_psnrs):.4f}")
        print(f"Training PSNR      (t1->t3, Δt=2.0): {np.mean(train_psnrs):.4f} +/- {np.std(train_psnrs):.4f}")
        print(f"Extrapolation PSNR (t1->t4, Δt=3.0): {np.mean(extrap_psnrs):.4f} +/- {np.std(extrap_psnrs):.4f}")
        print("="*60)
        
        # Save test set evaluation results to CSV
        test_results_data = []
        for i, result in enumerate(all_results):
            fold_idx = self.config["CV_FOLDS"][i]
            test_results_data.append({
                'fold': fold_idx,
                'model_path': os.path.basename(result['model_path']),
                'interpolation_t2_psnr': result['Interpolation_t2']['PSNR'],
                'interpolation_t2_ssim': result['Interpolation_t2']['SSIM'],
                'training_t3_psnr': result['Training_t3']['PSNR'],
                'training_t3_ssim': result['Training_t3']['SSIM'],
                'extrapolation_t4_psnr': result['Extrapolation_t4']['PSNR'],
                'extrapolation_t4_ssim': result['Extrapolation_t4']['SSIM']
            })
        
        test_results_df = pd.DataFrame(test_results_data)
        
        # Add summary statistics
        summary_row = {
            'fold': 'mean',
            'model_path': 'N/A',
            'interpolation_t2_psnr': np.mean(interp_psnrs),
            'interpolation_t2_ssim': np.mean([r['Interpolation_t2']['SSIM'] for r in all_results]),
            'training_t3_psnr': np.mean(train_psnrs),
            'training_t3_ssim': np.mean([r['Training_t3']['SSIM'] for r in all_results]),
            'extrapolation_t4_psnr': np.mean(extrap_psnrs),
            'extrapolation_t4_ssim': np.mean([r['Extrapolation_t4']['SSIM'] for r in all_results])
        }
        std_row = {
            'fold': 'std',
            'model_path': 'N/A',
            'interpolation_t2_psnr': np.std(interp_psnrs),
            'interpolation_t2_ssim': np.std([r['Interpolation_t2']['SSIM'] for r in all_results]),
            'training_t3_psnr': np.std(train_psnrs),
            'training_t3_ssim': np.std([r['Training_t3']['SSIM'] for r in all_results]),
            'extrapolation_t4_psnr': np.std(extrap_psnrs),
            'extrapolation_t4_ssim': np.std([r['Extrapolation_t4']['SSIM'] for r in all_results])
        }
        
        test_results_df = pd.concat([test_results_df, pd.DataFrame([summary_row, std_row])], ignore_index=True)
        
        test_results_csv_path = os.path.join(self.results_dir, "test_set_evaluation_results.csv")
        test_results_df.to_csv(test_results_csv_path, index=False)
        print(f"📊 Test set evaluation results saved to {test_results_csv_path}\n")
        
        best_result = max(all_results, key=lambda x: x['Training_t3']['PSNR'])
        best_model_name = os.path.basename(best_result['model_path']).split('.')[0]
        print(f"🏆 Best model: {best_model_name} (Training PSNR: {max(train_psnrs):.4f})")
        
        predicted_flair_dir = os.path.join(self.results_dir, f"{best_model_name}_Pred_Scan3Wave4_3D")
        ground_truth_wmh_dir = os.path.join(self.config["ROOT_DIR"], "Scan3Wave4_WMH")
        
        # ============================================================
        # === STAGE 2: WMH SEGMENTATION INFERENCE (PRETRAINED) =======
        # ============================================================
        
        print(f"\n{'='*60}")
        print("🔬 Starting Stage 2: WMH Segmentation Inference")
        print(f"{'='*60}")
        
        # Path to pretrained SwinUNETR model (same folder as this file)
        pretrained_model_path = os.path.join(
            self.models_dir, 
            "best_swin_segmentation_model.pth"
        )
        
        if not os.path.exists(pretrained_model_path):
            print(f"⚠️ Pretrained model not found at: {pretrained_model_path}")
            print("Stage 2 skipped. Please ensure the pretrained model exists.")
        elif not os.path.exists(predicted_flair_dir):
            print(f"⚠️ Predicted FLAIR directory not found: {predicted_flair_dir}")
            print("Stage 2 skipped.")
        elif not os.path.exists(ground_truth_wmh_dir):
            print(f"⚠️ Ground truth WMH directory not found: {ground_truth_wmh_dir}")
            print("Stage 2 skipped.")
        else:
            # Run Stage 2 inference using the pretrained model
            stage2_results = self.run_stage2_inference(
                pred_flair_dir=predicted_flair_dir,
                wmh_gt_dir=ground_truth_wmh_dir,
                pretrained_model_path=pretrained_model_path,
                time_point_label="Scan3Wave4"
            )
            
            if stage2_results:
                print(f"\n✅ Stage 2 completed successfully!")
                print(f"   Dice Score: {stage2_results['dice_score']:.4f}")
            else:
                print("\n⚠️ Stage 2 encountered errors.")
        
        print(f"\n{'='*60}")
        print(f"🎉 Experiment {self.experiment_number} Complete!")
        print(f"{'='*60}")
                
        return os.path.join(self.results_dir, f"{best_model_name}_Pred_Scan3Wave4"), ground_truth_wmh_dir
    
    def _stage2_wmh_segmentation(self, pred_flair_dir, wmh_gt_dir):
        """Run Stage 2: WMH Segmentation."""
        print("\n" + "="*60)
        print("Starting Stage 2 (WMH Segmentation)")
        print("="*60)
        
        predicted_flair_dir_3d = f"{pred_flair_dir}_3D"
        
        if not os.path.exists(predicted_flair_dir_3d):
            print(f"[Stage 2] Directory not found: {predicted_flair_dir_3d}")
            return
        
        if not os.path.exists(wmh_gt_dir):
            print(f"[Stage 2] Directory not found: {wmh_gt_dir}")
            return
        
        # Run segmentation
        run_stage2_segmentation(
            predicted_flair_dir_3d,
            wmh_gt_dir,
            self.config["DEVICE"],
            self.models_dir,
            self.config["NUM_EPOCHS"]
        )
        
        # Volume progression analysis
        print("\n" + "="*60)
        print("Performing WMH Volume Progression Analysis")
        print("="*60 + "\n")
        
        time_points = ['Scan1Wave2', 'Scan2Wave3', 'Scan3Wave4', 'Scan4Wave5']
        gt_wmh_dirs = {
            tp: os.path.join(self.config["ROOT_DIR"], f"{tp}_WMH")
            for tp in time_points
        }
        
        # Check for missing directories
        missing = [tp for tp, d in gt_wmh_dirs.items() if not os.path.exists(d)]
        if missing:
            print(f"⚠️ Missing directories for: {missing}")
            return
        
        # Analyze volumes
        volume_results = analyze_wmh_volume_progression(
            self.results_dir,
            self.models_dir,
            gt_wmh_dirs,
            time_points,
            self.config["DEVICE"]
        )
        
        if volume_results:
            plot_volume_progression(volume_results, self.get_plots_path("volume_progression.png"))
            
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
                print(f"   Mean Error: {np.mean(errors):.2f} +/- {np.std(errors):.2f} ml")
                print(f"   Min Error: {np.min(errors):.2f} ml")
                print(f"   Max Error: {np.max(errors):.2f} ml")


# ============================================================
# === STANDALONE EXECUTION ===
# ============================================================

if __name__ == "__main__":
    """
    Run this experiment directly without going through main.py
    Usage: python experiments/flair_to_flair_unet_baseline.py
    """
    print("\n" + "="*70)
    print("🧪 Running Experiment 9: FLAIR → FLAIR (UNet)")
    print("="*70 + "\n")
    
    # Import config from main.py (reuse the same config)
    from main import CONFIG as MAIN_CONFIG
    CONFIG = MAIN_CONFIG
    
    print(f"Using device: {CONFIG['DEVICE']}")
    
    # Experiment configuration
    experiment_config = {
        "name": "flair_to_flair_unet_baseline",
        "description": "FLAIR -> FLAIR (two-stage: prediction then segmentation, loss: L1 only)",
        "use_wmh": True,
        "class": Experiment9
    }
    
    # Run experiment
    experiment = Experiment9(
        experiment_number=9,
        experiment_config=experiment_config,
        config=CONFIG
    )
    experiment.run()
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70 + "\n")