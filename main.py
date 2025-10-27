# -*- coding: utf-8 -*-
"""
Main entry point for WMH Prediction experiments using ImageFlowNet.

This script provides a unified interface to run different experiments by simply
changing the EXPERIMENT_NUMBER variable. All experiments follow the same pipeline:
  1. FLAIR prediction using ImageFlowNet with cross-validation
  2. WMH segmentation from predicted FLAIR (Stage 2)
  3. Volume progression analysis

To run different experiments, just change EXPERIMENT_NUMBER at the top and run main.py.
"""

import os
import torch

# ============================================================
# === CONFIGURATION: Change EXPERIMENT_NUMBER to switch ===
# ============================================================

EXPERIMENT_NUMBER = 1  # Change this to run different experiments
                       # 1 = FLAIR → FLAIR (without WMH input)
                       # 2 = FLAIR+WMH → FLAIR (with WMH input)

# ============================================================
# === GLOBAL CONFIGURATION ===
# ============================================================

CONFIG = {
    # Dataset
    "ROOT_DIR": "/app/dataset/LBC1936",
    "FOLD_CSV": "patients_5fold.csv",
    
    # Training
    "BATCH_SIZE": 2,
    "LEARNING_RATE": 1e-4,
    "NUM_EPOCHS": 10,
    "MAX_SLICES": 48,
    "MAX_PATIENTS_PER_FOLD": 5,
    
    # Thresholds and coefficients
    "RECON_PSNR_THR": 25.0,
    "CONTRASTIVE_COEFF": 0.1,
    
    # Cross-validation
    "CV_FOLDS": [1, 2, 3, 4],
    "TEST_FOLD": 5,
    
    # Device
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Set derived values
CONFIG["K_FOLDS"] = len(CONFIG["CV_FOLDS"])

print(f"Using device: {CONFIG['DEVICE']}")

def main():
    """
    Main entry point for running experiments.
    Loads the selected experiment and executes it.
    """
    # Import experiment registry
    from experiments import EXPERIMENTS
    
    # Validate experiment number
    if EXPERIMENT_NUMBER not in EXPERIMENTS:
        raise ValueError(
            f"Invalid experiment number: {EXPERIMENT_NUMBER}. "
            f"Available experiments: {list(EXPERIMENTS.keys())}"
        )
    
    # Get experiment configuration
    exp_config = EXPERIMENTS[EXPERIMENT_NUMBER]
    experiment_class = exp_config["class"]
    
    # Print experiment information
    print("\n" + "="*70)
    print(f"🧪 EXPERIMENT {EXPERIMENT_NUMBER}: {exp_config['description']}")
    print("="*70)
    print(f"Name:        {exp_config['name']}")
    print(f"Use WMH:     {exp_config['use_wmh']}")
    print(f"Description: {exp_config['description']}")
    print("="*70 + "\n")
    
    # Initialize and run experiment
    experiment = experiment_class(EXPERIMENT_NUMBER, exp_config, CONFIG)
    experiment.run()
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()