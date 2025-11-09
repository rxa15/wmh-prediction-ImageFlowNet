# -*- coding: utf-8 -*-
"""
Experiments module for WMH Prediction using ImageFlowNet.
Contains modular experiment definitions that can be easily added and managed.
"""

from base import BaseExperiment
from flair_to_flair import Experiment1
from flair_to_flair_wmh import Experiment3

# Registry of available experiments
EXPERIMENTS = {
    1: {
        "name": "flair_to_flair",
        "use_wmh": False,
        "description": "FLAIR → FLAIR (two-stage: prediction then segmentation)",
        "class": Experiment1
    },
    3: {
        "name": "flair_to_flair_wmh",
        "use_wmh": True,
        "description": "FLAIR → FLAIR+WMH (joint training: single-stage prediction)",
        "class": Experiment3
    }
}

__all__ = ["EXPERIMENTS", "BaseExperiment", "Experiment1", "Experiment3"]
