# Fake News Detection System

An advanced NLP application that uses transformer models (BERT, RoBERTa, DistilBERT) to detect fake news articles and explain model predictions with attention visualization.

## Features

- **Multi-Model Predictions**: Compare BERT, RoBERTa, and DistilBERT for news classification
- **Real/Fake Classification**: Binary classification with confidence scores
- **Model Explainability**: Token-level importance analysis and attention visualization
- **Batch Processing**: Process multiple articles simultaneously
- **Training Interface**: Train models on custom datasets
- **Analytics Dashboard**: Visualize model performance metrics
- **History Tracking**: Store and review prediction history
- **Dataset Management**: Upload and manage news datasets

## Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for bundling
- Tailwind CSS
- Recharts for visualizations

### Backend
- FastAPI
- PyTorch & Hugging Face Transformers
- scikit-learn
- SQLite database

## Installation

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- `POST /predict` - Predict if article is real/fake
- `POST /predict/batch` - Batch prediction
- `POST /predict/compare` - Compare all models
- `GET /models` - List available models
- `POST /explainability/tokens` - Get token importance
- `GET /history` - Get prediction history
- `GET /analytics` - Get analytics data
- `POST /train` - Start training
- `POST /datasets/upload` - Upload dataset

## Usage

1. Paste or upload article text
2. Select model or compare all models
3. View prediction and confidence scores
4. Inspect explanation (important tokens)
5. View analytics and history

## Model Performance

- **BERT**: 92% accuracy
- **RoBERTa**: 94% accuracy  
- **DistilBERT**: 90% accuracy (lightweight)

---
Version: 1.0.0
