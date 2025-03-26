import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import json
from .base_model import BaseModel

class TextDataset(Dataset):
    """Dataset class for text data."""
    def __init__(self, texts, labels, word_to_idx, max_sequence_length):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_sequence_length = max_sequence_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to sequence of indices
        words = text.lower().split()
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                   for word in words[:self.max_sequence_length]]
        
        # Pad sequence if necessary
        if len(sequence) < self.max_sequence_length:
            sequence.extend([self.word_to_idx['<PAD>']] * 
                          (self.max_sequence_length - len(sequence)))
        
        return torch.tensor(sequence), torch.tensor(label)

class LSTMModel(nn.Module):
    """LSTM model for sentiment analysis."""
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Use last output
        out = self.fc(lstm_out)
        return out

class DeepLearningModel(BaseModel):
    """Deep learning model using LSTM."""
    
    def __init__(self, max_words=10000, max_sequence_length=100,
                 embedding_dim=100, lstm_units=64):
        """Initialize the model.
        
        Args:
            max_words (int): Maximum vocabulary size
            max_sequence_length (int): Maximum sequence length
            embedding_dim (int): Dimension of word embeddings
            lstm_units (int): Number of LSTM units
        """
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        # Initialize model components
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_vocabulary(self, texts):
        """Build vocabulary from texts."""
        words = []
        for text in texts:
            words.extend(text.lower().split())
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Add most frequent words to vocabulary
        for word, _ in word_counts.most_common(self.max_words - len(self.word_to_idx)):
            self.word_to_idx[word] = len(self.word_to_idx)
    
    def train(self, train_data, val_data=None, epochs=10, batch_size=32, **kwargs):
        """Train the model.
        
        Args:
            train_data (dict): Dictionary containing 'texts' and 'labels'
            val_data (dict, optional): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training history
        """
        # Build vocabulary
        self._build_vocabulary(train_data['texts'])
        
        # Create datasets
        train_dataset = TextDataset(
            train_data['texts'],
            train_data['labels'],
            self.word_to_idx,
            self.max_sequence_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Initialize model
        self.model = LSTMModel(
            vocab_size=len(self.word_to_idx),
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            num_classes=2
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        # Training loop
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_texts, batch_labels in train_loader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct / total
            
            history['train_loss'].append(epoch_loss)
            history['train_accuracy'].append(epoch_acc)
            
            # Validation
            if val_data is not None:
                val_loss, val_acc = self._evaluate(val_data, criterion)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        
        return history
    
    def _evaluate(self, data, criterion=None):
        """Evaluate the model on given data."""
        self.model.eval()
        dataset = TextDataset(
            data['texts'],
            data['labels'],
            self.word_to_idx,
            self.max_sequence_length
        )
        dataloader = DataLoader(dataset, batch_size=32)
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_texts)
                if criterion is not None:
                    loss = criterion(outputs, batch_labels)
                    total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total
        if criterion is not None:
            loss = total_loss / len(dataloader)
            return loss, accuracy
        return accuracy
    
    def predict(self, texts):
        """Make predictions on new texts.
        
        Args:
            texts (list): List of text strings to predict
            
        Returns:
            np.ndarray: Array of predictions (0 for negative, 1 for positive)
        """
        self.model.eval()
        dataset = TextDataset(
            texts,
            [0] * len(texts),  # Dummy labels
            self.word_to_idx,
            self.max_sequence_length
        )
        dataloader = DataLoader(dataset, batch_size=32)
        
        predictions = []
        with torch.no_grad():
            for batch_texts, _ in dataloader:
                batch_texts = batch_texts.to(self.device)
                outputs = self.model(batch_texts)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, test_data):
        """Evaluate the model on test data.
        
        Args:
            test_data (dict): Dictionary containing 'texts' and 'labels'
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        accuracy = self._evaluate(test_data)
        return {'accuracy': accuracy}
    
    def save(self, path):
        """Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'word_to_idx': self.word_to_idx,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units
        }
        torch.save(model_data, path)
    
    def load(self, path):
        """Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        model_data = torch.load(path)
        self.word_to_idx = model_data['word_to_idx']
        self.max_sequence_length = model_data['max_sequence_length']
        self.embedding_dim = model_data['embedding_dim']
        self.lstm_units = model_data['lstm_units']
        
        # Reinitialize model
        self.model = LSTMModel(
            vocab_size=len(self.word_to_idx),
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            num_classes=2
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(model_data['model_state_dict']) 