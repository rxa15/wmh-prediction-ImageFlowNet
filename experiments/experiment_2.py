# -*- coding: utf-8 -*-
"""
Experiment 2: FLAIR+WMH → FLAIR with downstream segmentation
"""

from .base import BaseExperiment
from .utils import (
    FLAIREvolutionDataset,
    LinearWarmupCosineAnnealingLR,
    neg_cos_sim,
    train_epoch,
    val_epoch,
    load_folds_from_csv,
    visualize_results,
    plot_fold_history,
    evaluate_and_visualize_tasks,
    run_stage2_segmentation,
    analyze_wmh_volume_progression,
    plot_volume_progression,
)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import pandas as pd
import numpy as np
from tqdm import tqdm
from ImageFlowNet.src.nn.imageflownet_ode import ImageFlowNetODE


class Experiment2(BaseExperiment):
    """
    Experiment 2: FLAIR+WMH → FLAIR prediction with downstream WMH segmentation.
    Uses both FLAIR and WMH as input channels.
    """
    
    def __init__(self, experiment_number, experiment_config, config):
        """
        Initialize Experiment 2.
        
        Args:
            experiment_number: The experiment number
            experiment_config: Dictionary with experiment configuration
            config: Global configuration dictionary
        """
        super().__init__(experiment_number, experiment_config)
        self.config = config
    
    def run(self):
        """Execute the full Experiment 2 pipeline."""
        print("\n" + "="*60)
        print("Starting Experiment 2: FLAIR+WMH → FLAIR")
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
        
        # Initialize dataset (with use_wmh=True for Experiment 2)
        print("Initializing dataset with WMH input channels...")
        full_dataset = FLAIREvolutionDataset(
            root_dir=self.config["ROOT_DIR"],
            max_slices_per_patient=self.config["MAX_SLICES"],
            use_wmh=self.use_wmh
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
        
        # Final Evaluation
        return self._evaluate_on_test_set(full_dataset, folds_dict)
    
    def _train_fold(self, val_fold_idx, full_dataset, folds_dict):
        """Train a model on a single fold."""
        print(f"\n{'='*50}")
        print(f"K-Fold Run: Validating on Fold {val_fold_idx}")
        print(f"{'='*50}\n")
        
        # Get patient splits
        val_pids = folds_dict[val_fold_idx][:self.config["MAX_PATIENTS_PER_FOLD"]]
        train_pids = [
            pid for f_idx in self.config["CV_FOLDS"] if f_idx != val_fold_idx
            for pid in folds_dict[f_idx]
        ][:self.config["MAX_PATIENTS_PER_FOLD"] * (self.config["K_FOLDS"] - 1)]
        
        print(f"Training patients:   {len(train_pids)}")
        print(f"Validation patients: {len(val_pids)}")
        
        # Create data indices
        train_indices = [i for i, item in enumerate(full_dataset.index_map) 
                        if item['patient_id'] in set(train_pids)]
        val_indices = [i for i, item in enumerate(full_dataset.index_map) 
                      if item['patient_id'] in set(val_pids)]
        
        # Create DataLoaders
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
        
        # Initialize model
        model = ImageFlowNetODE(
            device=self.config["DEVICE"],
            in_channels=2,
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
        
        best_val_psnr = 0.0
        recon_good_enough = False
        model_save_path = self.get_model_path(val_fold_idx)
        
        history = {
            'train_recon_loss': [],
            'train_pred_loss': [],
            'val_recon_psnr': [],
            'val_pred_psnr': []
        }
        
        # Training loop
        for epoch in range(self.config["NUM_EPOCHS"]):
            avg_recon_loss, avg_pred_loss = train_epoch(
                model, train_loader, optimizer, ema, mse_loss,
                self.config["DEVICE"], epoch, recon_good_enough,
                self.config["NUM_EPOCHS"], self.config["CONTRASTIVE_COEFF"]
            )
            
            with ema.average_parameters():
                val_recon_psnr, val_pred_psnr = val_epoch(model, val_loader, self.config["DEVICE"])
            
            history['train_recon_loss'].append(avg_recon_loss)
            history['train_pred_loss'].append(avg_pred_loss)
            history['val_recon_psnr'].append(val_recon_psnr)
            history['val_pred_psnr'].append(val_pred_psnr)
            
            print(f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']} (Fold {val_fold_idx}): Val PSNR={val_pred_psnr:.4f}")
            
            if not recon_good_enough and val_recon_psnr > self.config["RECON_PSNR_THR"]:
                recon_good_enough = True
                print("Reconstruction threshold reached. Starting ODE training.")
            
            if val_pred_psnr > best_val_psnr:
                best_val_psnr = val_pred_psnr
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Val PSNR improved. Model saved to {model_save_path}")
            
            scheduler.step()
        
        # Plot training history
        plot_fold_history(history, val_fold_idx, self.plots_dir)
    
    def _evaluate_on_test_set(self, full_dataset, folds_dict):
        """Evaluate all trained models on the held-out test set."""
        print("\n" + "="*60)
        print("✅ CV Training Complete. Starting Final Evaluation on Held-Out Test Set.")
        print("="*60)
        
        # Get test split
        test_pids = folds_dict[self.config["TEST_FOLD"]][:self.config["MAX_PATIENTS_PER_FOLD"]]
        test_indices = [i for i, item in enumerate(full_dataset.index_map)
                       if item['patient_id'] in set(test_pids)]
        
        print(f"Using {len(test_pids)} patients from Fold {self.config['TEST_FOLD']} for final testing.")
        
        # Create DataLoaders
        source_loader = DataLoader(
            Subset(full_dataset, [i for i in test_indices 
                                 if full_dataset.index_map[i]['scan_pair'] == "Scan1Wave2"]),
            batch_size=self.config["BATCH_SIZE"],
            shuffle=False
        )
        
        gt_loaders = {
            "Scan2Wave3": DataLoader(
                Subset(full_dataset, [i for i in test_indices
                                     if full_dataset.index_map[i]['scan_pair'] == "Scan2Wave3"]),
                batch_size=self.config["BATCH_SIZE"],
                shuffle=False
            ),
            "Scan3Wave4": DataLoader(
                Subset(full_dataset, [i for i in test_indices
                                     if full_dataset.index_map[i]['scan_pair'] == "Scan3Wave4"]),
                batch_size=self.config["BATCH_SIZE"],
                shuffle=False
            ),
            "Scan4Wave5": DataLoader(
                Subset(full_dataset, [i for i in test_indices
                                     if full_dataset.index_map[i]['scan_pair'] == "Scan4Wave5"]),
                batch_size=self.config["BATCH_SIZE"],
                shuffle=False
            ),
        }
        
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
            evaluate_and_visualize_tasks(
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
        
        interp_psnrs = [r['Interpolation_t1']['PSNR'] for r in all_results]
        pred_psnrs = [r['Prediction_t2']['PSNR'] for r in all_results]
        extrap_psnrs = [r['Extrapolation_t3']['PSNR'] for r in all_results]
        
        print(f"Interpolation PSNR (t=1->2): {np.mean(interp_psnrs):.4f} +/- {np.std(interp_psnrs):.4f}")
        print(f"Prediction PSNR    (t=1->3): {np.mean(pred_psnrs):.4f} +/- {np.std(pred_psnrs):.4f}")
        print(f"Extrapolation PSNR (t=1->4): {np.mean(extrap_psnrs):.4f} +/- {np.std(extrap_psnrs):.4f}")
        print("="*60)
        
        # Find best model
        best_result = max(all_results, key=lambda x: x['Prediction_t2']['PSNR'])
        best_model_name = os.path.basename(best_result['model_path']).split('.')[0]
        print(f"🏆 Best model: {best_model_name} (Prediction PSNR: {max(pred_psnrs):.4f})")
        
        predicted_flair_dir = os.path.join(self.results_dir, f"{best_model_name}_Pred_Scan3Wave4")
        ground_truth_wmh_dir = os.path.join(self.config["ROOT_DIR"], "Scan3Wave4_WMH")
        
        return predicted_flair_dir, ground_truth_wmh_dir
    
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
            self.models_dir
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
