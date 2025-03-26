import pandas as pd
import numpy as np
from pathlib import Path
from data.preprocessor import TextPreprocessor
from data.dataset import SentimentDataset
from models.traditional_ml import TraditionalMLModel
from models.lstm_model import LSTMModel
from visualization.plotting import ModelVisualizer

def test_models():
    """Test both traditional ML and LSTM models on the IMDB dataset."""
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("data/imdb_train.csv")
    test_df = pd.read_csv("data/imdb_test.csv")
    
    # Initialize components
    preprocessor = TextPreprocessor()
    dataset = SentimentDataset()
    visualizer = ModelVisualizer()
    
    # Preprocess data
    print("Preprocessing data...")
    train_texts = preprocessor.process(train_df['text'].values)
    test_texts = preprocessor.process(test_df['text'].values)
    
    # Prepare data for training
    train_data, val_data, test_data = dataset.prepare_data(
        train_texts,
        train_df['sentiment'].values,
        test_size=0.2,
        val_size=0.2
    )
    
    # Test Traditional ML Model
    print("\nTesting Traditional ML Model...")
    traditional_model = TraditionalMLModel(
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    # Train traditional model
    print("Training traditional model...")
    traditional_history = traditional_model.train(
        train_data,
        val_data,
        epochs=1  # For testing, we'll use just one epoch
    )
    
    # Evaluate traditional model
    print("Evaluating traditional model...")
    traditional_metrics = traditional_model.evaluate(test_data)
    print("Traditional Model Metrics:", traditional_metrics)
    
    # Test LSTM Model
    print("\nTesting LSTM Model...")
    lstm_model = LSTMModel(
        max_words=10000,
        max_sequence_length=100,
        embedding_dim=100,
        lstm_units=64
    )
    
    # Train LSTM model
    print("Training LSTM model...")
    lstm_history = lstm_model.train(
        train_data,
        val_data,
        epochs=1,  # For testing, we'll use just one epoch
        batch_size=32
    )
    
    # Evaluate LSTM model
    print("Evaluating LSTM model...")
    lstm_metrics = lstm_model.evaluate(test_data)
    print("LSTM Model Metrics:", lstm_metrics)
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Plot training history
    visualizer.plot_training_history(traditional_history)
    visualizer.plot_training_history(lstm_history)
    
    # Plot confusion matrices
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
    
    # Plot model comparison
    model_results = {
        'Traditional ML': traditional_metrics,
        'LSTM': lstm_metrics
    }
    visualizer.plot_model_comparison(model_results)
    
    # Save models
    print("\nSaving models...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    traditional_model.save(str(model_dir / "traditional_model.joblib"))
    lstm_model.save(str(model_dir / "lstm_model"))
    
    print("Testing completed!")

if __name__ == "__main__":
    test_models() 