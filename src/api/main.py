from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.traditional_ml import TraditionalMLModel
from models.lstm_model import DeepLearningModel

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using traditional ML and LSTM models",
    version="1.0.0"
)

# Initialize models
traditional_model = None
lstm_model = None

class TextInput(BaseModel):
    """Input schema for text analysis."""
    text: str

class BatchTextInput(BaseModel):
    """Input schema for batch text analysis."""
    texts: List[str]

class ModelResponse(BaseModel):
    """Response schema for model predictions."""
    sentiment: str
    confidence: float
    model: str

class BatchModelResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[ModelResponse]

class ModelInfo(BaseModel):
    """Response schema for model information."""
    model_name: str
    architecture: str
    parameters: Dict[str, any]
    metrics: Optional[Dict[str, float]]

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global traditional_model, lstm_model
    
    # Load traditional model
    traditional_model = TraditionalMLModel()
    traditional_model_path = Path("models/traditional_model.joblib")
    if traditional_model_path.exists():
        traditional_model.load(str(traditional_model_path))
    
    # Load LSTM model
    lstm_model = DeepLearningModel()
    lstm_model_path = Path("models/lstm_model")
    if lstm_model_path.exists():
        lstm_model.load(str(lstm_model_path))

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "traditional": traditional_model is not None,
            "lstm": lstm_model is not None
        }
    }

@app.post("/analyze/traditional", response_model=ModelResponse)
async def analyze_traditional(input_data: TextInput):
    """Analyze sentiment using the traditional ML model."""
    if traditional_model is None:
        raise HTTPException(status_code=503, detail="Traditional model not loaded")
    
    try:
        prediction = traditional_model.predict([input_data.text])[0]
        confidence = 0.8  # Placeholder for confidence score
        sentiment = "positive" if prediction == 1 else "negative"
        
        return ModelResponse(
            sentiment=sentiment,
            confidence=confidence,
            model="traditional"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/lstm", response_model=ModelResponse)
async def analyze_lstm(input_data: TextInput):
    """Analyze sentiment using the LSTM model."""
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")
    
    try:
        prediction = lstm_model.predict([input_data.text])[0]
        confidence = 0.8  # Placeholder for confidence score
        sentiment = "positive" if prediction == 1 else "negative"
        
        return ModelResponse(
            sentiment=sentiment,
            confidence=confidence,
            model="lstm"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch/traditional", response_model=BatchModelResponse)
async def analyze_batch_traditional(input_data: BatchTextInput):
    """Analyze sentiment for multiple texts using the traditional ML model."""
    if traditional_model is None:
        raise HTTPException(status_code=503, detail="Traditional model not loaded")
    
    try:
        predictions = traditional_model.predict(input_data.texts)
        results = []
        
        for pred in predictions:
            confidence = 0.8  # Placeholder for confidence score
            sentiment = "positive" if pred == 1 else "negative"
            results.append(ModelResponse(
                sentiment=sentiment,
                confidence=confidence,
                model="traditional"
            ))
        
        return BatchModelResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch/lstm", response_model=BatchModelResponse)
async def analyze_batch_lstm(input_data: BatchTextInput):
    """Analyze sentiment for multiple texts using the LSTM model."""
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")
    
    try:
        predictions = lstm_model.predict(input_data.texts)
        results = []
        
        for pred in predictions:
            confidence = 0.8  # Placeholder for confidence score
            sentiment = "positive" if pred == 1 else "negative"
            results.append(ModelResponse(
                sentiment=sentiment,
                confidence=confidence,
                model="lstm"
            ))
        
        return BatchModelResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info", response_model=List[ModelInfo])
async def get_model_info():
    """Get information about the loaded models."""
    models_info = []
    
    if traditional_model is not None:
        models_info.append(ModelInfo(
            model_name="Traditional ML",
            architecture="TF-IDF + Logistic Regression",
            parameters={
                "max_features": 5000,
                "ngram_range": (1, 2)
            },
            metrics=None  # Add actual metrics if available
        ))
    
    if lstm_model is not None:
        models_info.append(ModelInfo(
            model_name="LSTM",
            architecture="LSTM with Embeddings",
            parameters={
                "max_words": 10000,
                "max_sequence_length": 100,
                "embedding_dim": 100,
                "lstm_units": 64
            },
            metrics=None  # Add actual metrics if available
        ))
    
    return models_info 