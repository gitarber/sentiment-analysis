import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from typing import Dict, List, Union
import pandas as pd

class ModelVisualizer:
    """Class for visualizing model training and evaluation results."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 10)
    
    def plot_training_history(self, history):
        """Plot training and validation metrics over time.
        
        Args:
            history (dict): Dictionary containing training history
        """
        if not isinstance(history, dict) or not history:
            print("No valid training history to plot")
            return
            
        metrics = []
        if 'loss' in history:
            metrics.append(('loss', 'Loss'))
        if 'accuracy' in history:
            metrics.append(('accuracy', 'Accuracy'))
        if 'val_loss' in history:
            metrics.append(('val_loss', 'Validation Loss'))
        if 'val_accuracy' in history:
            metrics.append(('val_accuracy', 'Validation Accuracy'))
            
        if not metrics:
            print("No metrics found in training history")
            return
            
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
            
        for i, (metric_name, metric_label) in enumerate(metrics):
            axes[i].plot(history[metric_name], label=metric_label)
            axes[i].set_title(f'Model {metric_label}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric_label)
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Plot confusion matrix with enhanced visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum() * 100
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Use a better colormap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        
        # Add percentage annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, 
                        f'({cm_norm[i, j]:.1f}%)',
                        ha='center', va='center',
                        color='black' if cm_norm[i, j] < 50 else 'white')
        
        plt.title('Confusion Matrix', pad=20, fontsize=14)
        plt.xlabel('Predicted', labelpad=10)
        plt.ylabel('True', labelpad=10)
        
        # Add accuracy score
        accuracy = np.trace(cm) / np.sum(cm) * 100
        plt.text(1.0, -0.1, 
                f'Overall Accuracy: {accuracy:.1f}%',
                ha='center', va='center',
                transform=plt.gca().transAxes,
                fontsize=12)
        
        # Improve layout
        plt.tight_layout()
        plt.show()
    
    def plot_classification_report(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 labels: List[str] = None,
                                 figsize: tuple = (10, 6)) -> None:
        """
        Plot classification report as a heatmap.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str], optional): Label names
            figsize (tuple): Figure size
        """
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        
        # Convert report to DataFrame
        df_report = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=figsize)
        sns.heatmap(df_report.iloc[:-3, :].astype(float),
                   annot=True,
                   fmt='.2f',
                   cmap='YlOrRd')
        
        plt.title('Classification Report')
        plt.show()
    
    def plot_feature_importance(self, feature_importance, feature_names):
        """Plot feature importance scores.
        
        Args:
            feature_importance (np.ndarray): Feature importance scores
            feature_names (list): Feature names
        """
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.figure(figsize=(10, 6))
        plt.barh(pos, feature_importance[sorted_idx])
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.show()
    
    def plot_attention_weights(self, text, attention_weights, word_indices, word_to_idx):
        """Plot attention weights for each word in the text.
        
        Args:
            text (str): Input text
            attention_weights (np.ndarray): Attention weights
            word_indices (np.ndarray): Word indices
            word_to_idx (dict): Word to index mapping
        """
        # Convert indices back to words
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        words = [idx_to_word[idx] for idx in word_indices]
        
        # Create figure
        plt.figure(figsize=(12, 4))
        
        # Plot attention weights
        plt.bar(range(len(attention_weights)), attention_weights)
        plt.xticks(range(len(attention_weights)), words, rotation=45)
        plt.title('Attention Weights')
        plt.xlabel('Words')
        plt.ylabel('Attention Weight')
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, model_results):
        """Plot comparison of model metrics.
        
        Args:
            model_results (dict): Dictionary of model names and their metrics
        """
        # Extract metrics
        metrics = {}
        for model, result in model_results.items():
            for metric, value in result.items():
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append((model, value))
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for ax, (metric, values) in zip(axes, metrics.items()):
            models, scores = zip(*values)
            
            # Create bar plot
            bars = ax.bar(models, scores)
            
            # Customize appearance
            ax.set_title(f'{metric.title()} Comparison', pad=20)
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')
            
            # Rotate x-labels if they are long
            if max(len(model) for model in models) > 10:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show() 