# -*- coding: utf-8 -*-
"""
Main entry point for WMH Prediction experiments using ImageFlowNet.

This script provides a unified interface to run different experiments via command line.
All experiments follow the same pipeline with cross-validation.

Usage:
    python main.py --exp 1  # Run Experiment 1: FLAIR ? FLAIR (two-stage, loss: L1 only)
    python main.py --exp 2  # Run Experiment 2: FLAIR ? FLAIR (two-stage, loss: L1 + SSIM)
    
Available Experiments:
    1: FLAIR ? FLAIR (two-stage: prediction then segmentation, loss: L1 only)
    2: FLAIR ? FLAIR (two-stage: prediction then segmentation, loss: L1 + SSIM)
"""

import os
import torch
torch.cuda.empty_cache()
import argparse

# Import experiments directly (same folder)
from base import BaseExperiment
from flair_to_flair import Experiment1
from flair_to_flair_contrastive import Experiment2

# Registry of available experiments
EXPERIMENTS = {
    1: {
        "name": "flair_to_flair_baseline",
        "use_wmh": True,
        "description": "FLAIR -> FLAIR (two-stage: prediction then segmentation, loss: L1 only)",
        "class": Experiment1
    },
    2: {
        "name": "flair_to_flair_contrastive",
        "use_wmh": True,
        "description": "FLAIR -> FLAIR (two-stage: prediction then segmentation, loss: L1 + SSIM)",
        "class": Experiment2
    }
}

# ============================================================
# === GLOBAL CONFIGURATION ===
# ============================================================

CONFIG = {
    # Dataset
    "ROOT_DIR": "/app/dataset/LBC1936",
    # "ROOT_DIR": "/disk/febrian/Edinburgh_Data/LBC1936",
    "FOLD_CSV": "train_val_5fold.csv",
    "TEST_CSV": "test_set_patients.csv",
    
    # Training
    "BATCH_SIZE": 2,
    "LEARNING_RATE": 1e-4,
    "NUM_EPOCHS": 5,
    "MAX_SLICES": 8,
    "MAX_PATIENTS_PER_FOLD": 5,
    
    # Thresholds and coefficients
    "RECON_PSNR_THR": 40.0,
    "CONTRASTIVE_COEFF": 0.1,
    "SEG_LOSS_WEIGHT": 1.0,  # Weight for segmentation loss (Experiment 3)
    
    # Cross-validation
    "CV_FOLDS": [1, 2, 3, 4, 5],
    
    # Device
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Set derived values
CONFIG["K_FOLDS"] = len(CONFIG["CV_FOLDS"])

print(f"Using device: {CONFIG['DEVICE']}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run WMH Prediction experiments with ImageFlowNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Experiments:
  1: FLAIR ? FLAIR (two-stage, loss: L1 only)
  2: FLAIR ? FLAIR (two-stage, loss: L1 + SSIM)

Examples:
  python main.py --exp 1    # Run Experiment 1
  python main.py --exp 2    # Run Experiment 2
        """
    )
    parser.add_argument(
        '--exp',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help='Experiment number to run (1, 2, or 3)'
    )
    return parser.parse_args()

def main():
    """
    Main entry point for running experiments.
    Loads the selected experiment and executes it.
    """
    # Parse command line arguments
    args = parse_args()
    experiment_number = args.exp
    
    # Validate experiment number
    if experiment_number not in EXPERIMENTS:
        raise ValueError(
            f"Invalid experiment number: {experiment_number}. "
            f"Available experiments: {list(EXPERIMENTS.keys())}"
        )
    
    # Get experiment configuration
    exp_config = EXPERIMENTS[experiment_number]
    experiment_class = exp_config["class"]
    
    # Print experiment information
    print("\n" + "="*70)
    print(f"?? EXPERIMENT {experiment_number}: {exp_config['description']}")
    print("="*70)
    print(f"Name:        {exp_config['name']}")
    print(f"Use WMH:     {exp_config['use_wmh']}")
    print(f"Description: {exp_config['description']}")
    print("="*70 + "\n")
    
    # Initialize and run experiment
    experiment = experiment_class(experiment_number, exp_config, CONFIG)
    experiment.run()
    
    print("\n" + "="*70)
    print("? EXPERIMENT COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()