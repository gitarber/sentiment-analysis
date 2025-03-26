import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from models.traditional_ml import TraditionalMLModel
from models.lstm_model import DeepLearningModel
from visualization.plotting import ModelVisualizer

def load_data():
    """Load and prepare the IMDB dataset."""
    # Load data
    train_df = pd.read_csv("data/imdb_train.csv")
    test_df = pd.read_csv("data/imdb_test.csv")
    
    # Convert labels to numeric
    train_df['sentiment'] = train_df['sentiment'].map({'negative': 0, 'positive': 1})
    test_df['sentiment'] = test_df['sentiment'].map({'negative': 0, 'positive': 1})
    
    # Split training data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text'].values,
        train_df['sentiment'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Prepare data dictionaries
    train_data = {'texts': train_texts, 'labels': train_labels}
    val_data = {'texts': val_texts, 'labels': val_labels}
    test_data = {'texts': test_df['text'].values, 'labels': test_df['sentiment'].values}
    
    return train_data, val_data, test_data

def train_and_evaluate():
    """Train and evaluate both models."""
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data()
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Train and evaluate traditional model
    print("\nTraining traditional model...")
    traditional_model = TraditionalMLModel(
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    try:
        traditional_history = traditional_model.train(
            train_data,
            val_data,
            epochs=1  # For testing, we'll use just one epoch
        )
    except Exception as e:
        print(f"Warning: Could not get training history for traditional model: {e}")
        traditional_history = None
    
    print("\nEvaluating traditional model...")
    traditional_metrics = traditional_model.evaluate(test_data)
    print("Traditional Model Metrics:", traditional_metrics)
    
    # Train and evaluate LSTM model
    print("\nTraining LSTM model...")
    lstm_model = DeepLearningModel(
        max_words=10000,
        max_sequence_length=100,
        embedding_dim=100,
        lstm_units=64
    )
    
    lstm_history = lstm_model.train(
        train_data,
        val_data,
        epochs=1,  # For testing, we'll use just one epoch
        batch_size=32
    )
    
    print("\nEvaluating LSTM model...")
    lstm_metrics = lstm_model.evaluate(test_data)
    print("LSTM Model Metrics:", lstm_metrics)
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Plot confusion matrices
    print("Generating confusion matrices...")
    traditional_pred = traditional_model.predict(test_data['texts'])
    lstm_pred = lstm_model.predict(test_data['texts'])
    
    visualizer.plot_confusion_matrix(
        test_data['labels'],
        traditional_pred,
        labels=['negative', 'positive']
    )
    
    visualizer.plot_confusion_matrix(
        test_data['labels'],
        lstm_pred,
        labels=['negative', 'positive']
    )
    
    # Plot model comparison using metrics
    print("Generating model comparison...")
    model_results = {
        'Traditional ML': traditional_metrics,
        'LSTM': lstm_metrics
    }
    visualizer.plot_model_comparison(model_results)
    
    # Only plot LSTM training history if available
    if lstm_history and isinstance(lstm_history, dict) and len(lstm_history) > 0:
        print("Generating LSTM training history plot...")
        visualizer.plot_training_history(lstm_history)
    
    # Save models
    print("\nSaving models...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    traditional_model.save(str(model_dir / "traditional_model.joblib"))
    lstm_model.save(str(model_dir / "lstm_model"))
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    train_and_evaluate() 