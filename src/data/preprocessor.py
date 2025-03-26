import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Union
import numpy as np

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with NLTK resources."""
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Get stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing noise and normalizing.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into words.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from the token list.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: List of tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize the tokens to their base form.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Process the input text through all preprocessing steps.
        
        Args:
            text (Union[str, List[str]]): Input text or list of texts
            
        Returns:
            Union[str, List[str]]: Processed text or list of processed texts
        """
        if isinstance(text, str):
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Tokenize
            tokens = self.tokenize(cleaned_text)
            
            # Remove stopwords
            tokens = self.remove_stopwords(tokens)
            
            # Lemmatize
            tokens = self.lemmatize(tokens)
            
            # Join tokens back into text
            return ' '.join(tokens)
        
        elif isinstance(text, list):
            return [self.process(t) for t in text]
        
        else:
            raise TypeError("Input must be a string or list of strings")
    
    def create_vocabulary(self, texts: List[str], max_vocab_size: int = 10000) -> dict:
        """
        Create a vocabulary from a list of processed texts.
        
        Args:
            texts (List[str]): List of processed texts
            max_vocab_size (int): Maximum size of the vocabulary
            
        Returns:
            dict: Word to index mapping
        """
        # Count word frequencies
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create vocabulary
        vocabulary = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        # Add most frequent words to vocabulary
        for word, _ in sorted_words[:max_vocab_size - len(vocabulary)]:
            vocabulary[word] = len(vocabulary)
        
        return vocabulary
    
    def text_to_sequence(self, text: str, vocabulary: dict) -> List[int]:
        """
        Convert text to sequence of indices using vocabulary.
        
        Args:
            text (str): Input text
            vocabulary (dict): Word to index mapping
            
        Returns:
            List[int]: Sequence of indices
        """
        words = text.split()
        return [vocabulary.get(word, vocabulary['<UNK>']) for word in words] 