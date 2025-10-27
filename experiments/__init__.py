# -*- coding: utf-8 -*-
"""
Experiments module for WMH Prediction using ImageFlowNet.
Contains modular experiment definitions that can be easily added and managed.
"""

from .base import BaseExperiment
from .experiment_1 import Experiment1
from .experiment_2 import Experiment2

# Registry of available experiments
EXPERIMENTS = {
    1: {
        "name": "flair_to_flair",
        "use_wmh": False,
        "description": "FLAIR → FLAIR with downstream",
        "class": Experiment1
    },
    2: {
        "name": "flair_wmh_to_flair",
        "use_wmh": True,
        "description": "FLAIR+WMH → FLAIR with downstream",
        "class": Experiment2
    }
}

__all__ = ["EXPERIMENTS", "BaseExperiment", "Experiment1", "Experiment2"]
