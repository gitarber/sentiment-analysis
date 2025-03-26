import requests
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tarfile
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def download_imdb_data():
    """Download and prepare the IMDB dataset."""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download IMDB dataset
    print("Downloading IMDB dataset...")
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset_path = data_dir / "aclImdb_v1.tar.gz"
    
    if not dataset_path.exists():
        download_file(url, dataset_path)
    
    # Extract dataset
    print("Extracting dataset...")
    with tarfile.open(dataset_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    
    # Process the dataset
    imdb_dir = data_dir / "aclImdb"
    
    def read_text_files(directory):
        texts = []
        labels = []
        for label in ['pos', 'neg']:
            label_dir = directory / label
            for filename in label_dir.glob('*.txt'):
                with open(filename, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
        return texts, labels
    
    # Read training data
    print("Processing training data...")
    train_texts, train_labels = read_text_files(imdb_dir / 'train')
    
    # Read test data
    print("Processing test data...")
    test_texts, test_labels = read_text_files(imdb_dir / 'test')
    
    # Convert to DataFrame
    train_df = pd.DataFrame({
        'text': train_texts,
        'sentiment': train_labels
    })
    test_df = pd.DataFrame({
        'text': test_texts,
        'sentiment': test_labels
    })
    
    # Convert sentiment labels
    train_df['sentiment'] = train_df['sentiment'].map({0: 'negative', 1: 'positive'})
    test_df['sentiment'] = test_df['sentiment'].map({0: 'negative', 1: 'positive'})
    
    # Save datasets
    train_df.to_csv(data_dir / "imdb_train.csv", index=False)
    test_df.to_csv(data_dir / "imdb_test.csv", index=False)
    
    # Clean up
    dataset_path.unlink()  # Remove the tar.gz file
    
    print(f"Dataset saved to {data_dir}")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, test_df

if __name__ == "__main__":
    download_imdb_data() 