# -*- coding: utf-8 -*-
"""
Base experiment class that defines the common interface for all experiments.
"""

from abc import ABC, abstractmethod
import os


class BaseExperiment(ABC):
    """
    Abstract base class for all experiments.
    Defines the common interface and provides shared utilities.
    """
    
    def __init__(self, experiment_number, experiment_config):
        """
        Initialize the base experiment.
        
        Args:
            experiment_number: The experiment number
            experiment_config: Dictionary with experiment configuration
        """
        self.experiment_number = experiment_number
        self.name = experiment_config["name"]
        self.use_wmh = experiment_config["use_wmh"]
        self.description = experiment_config["description"]
        
        # Create experiment-specific directories
        self.results_dir = f"{self.name}_results"
        self.models_dir = f"{self.name}_models"
        self.plots_dir = f"{self.name}_plots"
        
        self._create_directories()
        self._print_info()
    
    def _create_directories(self):
        """Create experiment-specific output directories."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def _print_info(self):
        """Print experiment information."""
        print(f"🧪 Running Experiment {self.experiment_number}: {self.description}")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📁 Plots directory: {self.plots_dir}")
        print(f"🔧 Configuration: use_wmh = {self.use_wmh}")
    
    @abstractmethod
    def run(self):
        """
        Execute the experiment.
        This method must be implemented by subclasses.
        """
        pass
    
    def get_model_path(self, fold_idx):
        """Get the path for saving/loading a model for a specific fold."""
        return os.path.join(self.models_dir, f"model_fold_{fold_idx}.pth")
    
    def get_results_path(self, filename):
        """Get the path for saving results."""
        return os.path.join(self.results_dir, filename)
    
    def get_plots_path(self, filename):
        """Get the path for saving plots."""
        return os.path.join(self.plots_dir, filename)
