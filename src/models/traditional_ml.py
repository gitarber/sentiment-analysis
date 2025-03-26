import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from typing import Dict, List, Union

from .base_model import BaseModel

class TraditionalMLModel(BaseModel):
    """Traditional machine learning model for sentiment analysis using TF-IDF and Logistic Regression."""
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: tuple = (1, 2),
                 random_state: int = 42):
        """
        Initialize the traditional ML model.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            ngram_range (tuple): Range of n-gram features to consider
            random_state (int): Random seed for reproducibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        
        # Initialize vectorizer and model
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            multi_class='multinomial'
        )
        
    def train(self, 
              train_data: Dict[str, np.ndarray],
              val_data: Dict[str, np.ndarray] = None,
              **kwargs) -> Dict[str, List[float]]:
        """
        Train the model on the provided data.
        
        Args:
            train_data (Dict[str, np.ndarray]): Training data
            val_data (Dict[str, np.ndarray], optional): Validation data
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Fit vectorizer on training data
        X_train = self.vectorizer.fit_transform(train_data['texts'])
        y_train = train_data['labels']
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred, average='weighted'),
            'train_recall': recall_score(y_train, train_pred, average='weighted'),
            'train_f1': f1_score(y_train, train_pred, average='weighted')
        }
        
        # Calculate validation metrics if validation data is provided
        val_metrics = {}
        if val_data is not None:
            X_val = self.vectorizer.transform(val_data['texts'])
            y_val = val_data['labels']
            val_pred = self.model.predict(X_val)
            
            val_metrics = {
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_precision': precision_score(y_val, val_pred, average='weighted'),
                'val_recall': recall_score(y_val, val_pred, average='weighted'),
                'val_f1': f1_score(y_val, val_pred, average='weighted')
            }
        
        # Combine metrics
        history = {**train_metrics, **val_metrics}
        return {k: [v] for k, v in history.items()}
    
    def predict(self, texts: Union[str, List[str], np.ndarray]) -> np.ndarray:
        """
        Make predictions on the input texts.
        
        Args:
            texts (Union[str, List[str], np.ndarray]): Input texts
            
        Returns:
            np.ndarray: Predicted labels
        """
        texts = self.preprocess_input(texts)
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        return self.postprocess_output(predictions)
    
    def evaluate(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on the test data.
        
        Args:
            test_data (Dict[str, np.ndarray]): Test data
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        X_test = self.vectorizer.transform(test_data['texts'])
        y_test = test_data['labels']
        y_pred = self.model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get the most important features for each class.
        
        Args:
            top_n (int): Number of top features to return per class
            
        Returns:
            Dict[str, float]: Dictionary mapping class labels to feature importance
        """
        feature_names = self.vectorizer.get_feature_names_out()
        importance_dict = {}
        
        for i, class_name in enumerate(self.model.classes_):
            # Get feature importance for this class
            importance = self.model.coef_[i]
            # Get indices of top features
            top_indices = np.argsort(importance)[-top_n:][::-1]
            
            # Create dictionary of feature names and their importance scores
            feature_importance = {
                feature_names[idx]: importance[idx]
                for idx in top_indices
            }
            
            importance_dict[class_name] = feature_importance
        
        return importance_dict 