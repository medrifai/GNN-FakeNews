import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class MetricsCollector:
    """
    A class to collect, compute, and store various performance metrics for model evaluation.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.train_metrics = []
        self.val_metrics = []
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.predictions = []
        self.probabilities = []
        self.labels = []
    
    def add_metrics(self, train_loss, train_acc, train_auc, val_loss, val_acc=None, val_auc=None):
        """
        Add training and validation metrics for one epoch.
        Args:
            train_loss (float): Training loss
            train_acc (float): Training accuracy
            train_auc (float): Training AUC
            val_loss (float): Validation loss
            val_acc (float): Validation accuracy (optional)
            val_auc (float): Validation AUC (optional)
        """
        self.train_metrics.append({
            'loss': train_loss,
            'accuracy': train_acc,
            'auc': train_auc
        })

        self.val_metrics.append({
            'loss': val_loss,
            'accuracy': val_acc ,
            'auc': val_auc
        })
    
    def save_metrics(self, filepath):
        """
        Save collected metrics to a file.
        
        Args:
            filepath (str): Path where to save the metrics
        """
        metrics_dict = {
            'train': self.train_metrics,
            'validation': self.val_metrics
        }
        np.save(filepath, metrics_dict)
    
    def load_metrics(self, filepath):
        """
        Load metrics from a file.
        
        Args:
            filepath (str): Path to the file containing metrics
        """
        try:
            loaded_metrics = np.load(filepath, allow_pickle=True).item()
            if isinstance(loaded_metrics, dict):
                if 'train' in loaded_metrics:
                    self.train_metrics = loaded_metrics['train']
                if 'validation' in loaded_metrics:
                    self.val_metrics = loaded_metrics['validation']
            else:
                raise ValueError(f"File at {filepath} does not contain a valid metrics dictionary.")
        except Exception as e:
            print(f"Error loading metrics: {e}")
    
    def get_latest_metrics(self):
        """
        Get the most recent metrics.
        
        Returns:
            tuple: (train_metrics, val_metrics) for the last epoch
        """
        if not self.train_metrics or not self.val_metrics:
            return None, None
        
        return self.train_metrics[-1], self.val_metrics[-1]