# -*- coding: utf-8 -*-
"""
Experiment 1: FLAIR → FLAIR with downstream segmentation
"""

from base import BaseExperiment
from utils import (
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


class Experiment1(BaseExperiment):
    """
    Experiment 1: FLAIR → FLAIR prediction with downstream WMH segmentation.
    Uses only FLAIR data (no WMH input).
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

        # Load test set patient IDs
        test_csv = self.config["TEST_CSV", None]
        if test_csv is None or not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test CSV not found at {test_csv}")
        
        self.test_patient_ids = self._load_patient_ids(test_csv, column="patient_ID")
        print(f"Loaded {len(self.test_patient_ids)} explicit test patients from {test_csv}")
    
    def _load_patient_ids(self, csv_path, column="patient_ID"):
        df = pd.read_csv(csv_path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {csv_path}. Columns: {list(df.columns)}")
        return [str(x).strip() for x in df[column].astype(str).tolist()]
    
    def run(self):
        """Execute the full Experiment 1 pipeline."""
        print("\n" + "="*60)
        print("Starting Experiment 1: FLAIR → FLAIR")
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
            
            with torch.no_grad(), ema.average_parameters():
                # Sample training set for faster evaluation
                train_sample_loader = DataLoader(
                    Subset(full_dataset, train_indices[:len(val_indices)]),  # Same size as val
                    batch_size=self.config["BATCH_SIZE"],
                    shuffle=False
                )
                train_recon_psnr, train_pred_psnr = val_epoch(model, train_sample_loader, self.config["DEVICE"])

            history['train_recon_loss'].append(avg_recon_loss)
            history['train_pred_loss'].append(avg_pred_loss)
            history['train_recon_psnr'].append(train_recon_psnr)  # ✅ New
            history['train_pred_psnr'].append(train_pred_psnr)
            history['val_recon_psnr'].append(val_recon_psnr)
            history['val_pred_psnr'].append(val_pred_psnr)
            
            print(f"Epoch {epoch+1}: Train PSNR={train_pred_psnr:.4f}, Val PSNR={val_pred_psnr:.4f}")
            
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
        
        # Save training history to CSV
        history_df = pd.DataFrame({
            'epoch': range(1, len(history['train_recon_loss']) + 1),
            'train_recon_loss': history['train_recon_loss'],
            'train_pred_loss': history['train_pred_loss'],
            'train_recon_psnr': history['train_recon_psnr'],
            'train_pred_psnr': history['train_pred_psnr'],
            'val_recon_psnr': history['val_recon_psnr'],
            'val_pred_psnr': history['val_pred_psnr']
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
            os.path.dirname(__file__), 
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


# ============================================================
# === STANDALONE EXECUTION ===
# ============================================================

if __name__ == "__main__":
    """
    Run this experiment directly without going through main.py
    Usage: python -m experiments.flair_to_flair
    """
    print("\n" + "="*70)
    print("🧪 Running Experiment 1: FLAIR → FLAIR (Standalone Mode)")
    print("="*70 + "\n")
    
    # Configuration
    CONFIG = {
        # Dataset
        "ROOT_DIR": "/app/dataset/LBC1936",
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
        
        # Cross-validation
        "CV_FOLDS": [1, 2, 3, 4, 5],
        
        # Device
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    CONFIG["K_FOLDS"] = len(CONFIG["CV_FOLDS"])
    
    print(f"Using device: {CONFIG['DEVICE']}")
    
    # Experiment configuration
    experiment_config = {
        "name": "flair_to_flair",
        "description": "FLAIR → FLAIR prediction without WMH input",
        "use_wmh": False,
        "class": Experiment1
    }
    
    # Run experiment
    experiment = Experiment1(
        experiment_number=1,
        experiment_config=experiment_config,
        config=CONFIG
    )
    experiment.run()
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70 + "\n")
