from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Union

class BaseModel(ABC):
    """Base class for all sentiment analysis models."""
    
    @abstractmethod
    def train(self, train_data, val_data=None, **kwargs):
        """Train the model on the given data.
        
        Args:
            train_data (dict): Dictionary containing 'texts' and 'labels'
            val_data (dict, optional): Validation data with same structure as train_data
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training history containing metrics
        """
        pass
    
    @abstractmethod
    def predict(self, texts):
        """Make predictions on new texts.
        
        Args:
            texts (list): List of text strings to predict
            
        Returns:
            np.ndarray: Array of predictions (0 for negative, 1 for positive)
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data):
        """Evaluate the model on test data.
        
        Args:
            test_data (dict): Dictionary containing 'texts' and 'labels'
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        pass
    
    def preprocess_input(self, texts: Union[str, List[str], np.ndarray]) -> np.ndarray:
        """
        Preprocess the input texts.
        
        Args:
            texts (Union[str, List[str], np.ndarray]): Input texts
            
        Returns:
            np.ndarray: Preprocessed texts
        """
        if isinstance(texts, str):
            texts = [texts]
        return np.array(texts)
    
    def postprocess_output(self, predictions: np.ndarray) -> np.ndarray:
        """
        Postprocess the model predictions.
        
        Args:
            predictions (np.ndarray): Raw model predictions
            
        Returns:
            np.ndarray: Processed predictions
        """
        return predictions 