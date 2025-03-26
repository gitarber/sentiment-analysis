import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SentimentDataset:
    def __init__(self, data_path: str = None):
        """
        Initialize the sentiment dataset handler.
        
        Args:
            data_path (str, optional): Path to the dataset file
        """
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        
    def load_imdb_data(self) -> pd.DataFrame:
        """
        Load the IMDB movie reviews dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        # For demonstration, we'll create a small sample dataset
        # In practice, you would load the actual IMDB dataset
        data = {
            'text': [
                "This movie was absolutely fantastic!",
                "I really enjoyed watching this film.",
                "It was okay, but nothing special.",
                "I didn't like this movie at all.",
                "Terrible waste of time."
            ],
            'sentiment': ['positive', 'positive', 'neutral', 'negative', 'negative']
        }
        return pd.DataFrame(data)
    
    def load_twitter_data(self) -> pd.DataFrame:
        """
        Load Twitter sentiment dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        # For demonstration, we'll create a small sample dataset
        # In practice, you would load the actual Twitter dataset
        data = {
            'text': [
                "Love this new product! #amazing",
                "Great service, highly recommended!",
                "Not sure about this one...",
                "This is the worst experience ever!",
                "Disappointed with the quality."
            ],
            'sentiment': ['positive', 'positive', 'neutral', 'negative', 'negative']
        }
        return pd.DataFrame(data)
    
    def prepare_data(self, 
                    texts: List[str], 
                    labels: List[str],
                    test_size: float = 0.2,
                    val_size: float = 0.2,
                    random_state: int = 42) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare the dataset for training.
        
        Args:
            texts (List[str]): List of text samples
            labels (List[str]): List of sentiment labels
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of remaining data to use for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Training and test data
        """
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels, test_size=test_size, random_state=random_state
        )
        
        # Split training set into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state
        )
        
        # Create data dictionaries
        train_data = {
            'texts': np.array(X_train),
            'labels': y_train
        }
        
        val_data = {
            'texts': np.array(X_val),
            'labels': y_val
        }
        
        test_data = {
            'texts': np.array(X_test),
            'labels': y_test
        }
        
        return train_data, val_data, test_data
    
    def create_data_generator(self, 
                            texts: np.ndarray, 
                            labels: np.ndarray,
                            batch_size: int = 32,
                            shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a data generator for batch processing.
        
        Args:
            texts (np.ndarray): Array of text samples
            labels (np.ndarray): Array of labels
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Batch of texts and labels
        """
        num_samples = len(texts)
        
        if shuffle:
            indices = np.random.permutation(num_samples)
            texts = texts[indices]
            labels = labels[indices]
        
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield batch_texts, batch_labels
    
    def get_label_mapping(self) -> Dict[int, str]:
        """
        Get the mapping between encoded labels and original labels.
        
        Returns:
            Dict[int, str]: Mapping between encoded and original labels
        """
        return dict(zip(range(len(self.label_encoder.classes_)), 
                       self.label_encoder.classes_))
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        Decode encoded labels back to original labels.
        
        Args:
            encoded_labels (np.ndarray): Array of encoded labels
            
        Returns:
            List[str]: List of decoded labels
        """
        return self.label_encoder.inverse_transform(encoded_labels) 