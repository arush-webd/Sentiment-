import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import pipeline, AutoModelForSequenceClassification
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class AdvancedFinancialSentimentAnalyzer:
    """
    Advanced multi-model sentiment analysis system for financial text
    Combines multiple approaches for robust sentiment scoring
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.tokenizers = {}
        self.financial_lexicon = self._build_financial_lexicon()
        self.indian_market_terms = self._build_indian_market_terms()
        
        # Initialize models
        self._load_models()
        
    def _build_financial_lexicon(self):
        """Build custom financial sentiment lexicon"""
        financial_positive = {
            'bullish': 2.0, 'rally': 1.5, 'surge': 2.0, 'soar': 2.0, 'breakout': 1.5,
            'outperform': 1.5, 'upgrade': 1.8, 'beat': 1.5, 'exceed': 1.5, 'strong': 1.2,
            'robust': 1.3, 'solid': 1.2, 'growth': 1.3, 'profit': 1.5, 'gain': 1.3,
            'positive': 1.0, 'optimistic': 1.2, 'confident': 1.2, 'expansion': 1.3,
            'recovery': 1.4, 'improvement': 1.3, 'dividend': 1.2, 'buyback': 1.4,
            'merger': 1.1, 'acquisition': 1.1, 'partnership': 1.0, 'deal': 1.0,
            'revenue': 1.1, 'earnings': 1.1, 'milestone': 1.3, 'breakthrough': 1.6
        }
        
        financial_negative = {
            'bearish': -2.0, 'crash': -2.5, 'plunge': -2.0, 'tumble': -1.8, 'decline': -1.3,
            'fall': -1.2, 'drop': -1.2, 'underperform': -1.5, 'downgrade': -1.8,
            'miss': -1.5, 'weak': -1.2, 'poor': -1.3, 'loss': -1.5, 'deficit': -1.4,
            'negative': -1.0, 'pessimistic': -1.2, 'concern': -1.1, 'worry': -1.2,
            'risk': -1.0, 'uncertainty': -1.1, 'volatility': -1.0, 'correction': -1.3,
            'recession': -2.0, 'crisis': -2.2, 'bankruptcy': -2.5, 'default': -2.3,
            'lawsuit': -1.5, 'fraud': -2.0, 'scandal': -2.0, 'investigation': -1.4,
            'layoff': -1.6, 'restructuring': -1.2, 'debt': -1.1, 'writedown': -1.5
        }
        
        return {**financial_positive, **{k: v for k, v in financial_negative.items()}}
    
    def _build_indian_market_terms(self):
        """Build Indian market specific terms"""
        return {
            'nifty': 1.0, 'sensex': 1.0, 'bse': 1.0, 'nse': 1.0,
            'sebi': 0.5, 'rbi': 0.8, 'fii': 0.7, 'dii': 0.7,
            'rupee': 0.5, 'inflation': -0.5, 'gdp': 0.8, 'fiscal': 0.3,
            'monsoon': 0.5, 'budget': 0.5, 'policy': 0.3, 'reform': 0.8,
            'digitalization': 1.2, 'startup': 1.0, 'unicorn': 1.5,
            'pli': 1.0, 'make_in_india': 1.0, 'atmanirbhar': 1.0
        }
    
    def _load_models(self):
        """Load pre-trained models"""
        print("Loading pre-trained models...")
        
        # 1. FinBERT for financial sentiment
        try:
            self.models['finbert'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if self.device == 'cuda' else -1
            )
            print("✓ FinBERT loaded successfully")
        except Exception as e:
            print(f"✗ FinBERT loading failed: {e}")
            
        # 2. VADER for rule-based sentiment
        self.models['vader'] = SentimentIntensityAnalyzer()
        print("✓ VADER loaded successfully")
        
        # 3. General BERT for broader context
        try:
            self.models['bert'] = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if self.device == 'cuda' else -1
            )
            print("✓ General BERT loaded successfully")
        except Exception as e:
            print(f"✗ General BERT loading failed: {e}")
    
    def preprocess_text(self, text):
        """Advanced text preprocessing for financial content"""
        if not isinstance(text, str):
            return ""
        
        # Clean text
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        # Handle financial numbers and percentages
        text = re.sub(r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', r'INR \1', text)
        text = re.sub(r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', r'USD \1', text)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
        text = re.sub(r'(\d+(?:,\d+)*)\s*crore', r'\1 crore', text)
        text = re.sub(r'(\d+(?:,\d+)*)\s*lakh', r'\1 lakh', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_financial_entities(self, text):
        """Extract financial entities and their context"""
        entities = {
            'companies': [],
            'financial_metrics': [],
            'numbers': [],
            'time_periods': []
        }
        
        # Extract company mentions (basic approach)
        company_pattern = r'\b[A-Z][a-z]+\s+(?:Ltd|Limited|Corp|Corporation|Inc|Company)\b'
        entities['companies'] = re.findall(company_pattern, text)
        
        # Extract financial metrics
        metric_patterns = [
            r'\b(?:revenue|profit|loss|earnings|ebitda|margin|growth)\b',
            r'\b(?:market cap|pe ratio|debt|equity|roe|roa)\b'
        ]
        for pattern in metric_patterns:
            entities['financial_metrics'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract numbers with context
        number_pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crore|lakh|million|billion|percent|%)'
        entities['numbers'] = re.findall(number_pattern, text)
        
        return entities
    
    def lexicon_based_sentiment(self, text):
        """Custom lexicon-based sentiment scoring"""
        words = text.lower().split()
        score = 0
        word_count = 0
        
        for word in words:
            if word in self.financial_lexicon:
                score += self.financial_lexicon[word]
                word_count += 1
            elif word in self.indian_market_terms:
                score += self.indian_market_terms[word] * 0.5
                word_count += 1
        
        if word_count == 0:
            return 0
        
        return score / word_count
    
    def analyze_sentiment(self, text, method='ensemble'):
        """
        Comprehensive sentiment analysis using multiple methods
        """
        if not text or not isinstance(text, str):
            return {'compound': 0, 'confidence': 0, 'method': 'empty'}
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        if method == 'ensemble':
            return self._ensemble_sentiment(clean_text)
        elif method == 'finbert':
            return self._finbert_sentiment(clean_text)
        elif method == 'vader':
            return self._vader_sentiment(clean_text)
        elif method == 'lexicon':
            return self._lexicon_sentiment(clean_text)
        else:
            return self._ensemble_sentiment(clean_text)
    
    def _ensemble_sentiment(self, text):
        """Ensemble method combining multiple sentiment approaches"""
        scores = []
        confidences = []
        
        # 1. FinBERT sentiment
        if 'finbert' in self.models:
            try:
                finbert_result = self.models['finbert'](text)[0]
                finbert_score = finbert_result['score'] if finbert_result['label'] == 'positive' else -finbert_result['score']
                if finbert_result['label'] == 'neutral':
                    finbert_score = 0
                scores.append(finbert_score)
                confidences.append(finbert_result['score'])
            except Exception as e:
                print(f"FinBERT error: {e}")
        
        # 2. VADER sentiment
        vader_result = self.models['vader'].polarity_scores(text)
        scores.append(vader_result['compound'])
        confidences.append(abs(vader_result['compound']))
        
        # 3. Custom lexicon sentiment
        lexicon_score = self.lexicon_based_sentiment(text)
        scores.append(np.tanh(lexicon_score))  # Normalize to [-1, 1]
        confidences.append(min(abs(lexicon_score), 1.0))
        
        # 4. General BERT (if available)
        if 'bert' in self.models:
            try:
                bert_result = self.models['bert'](text)[0]
                # Convert to sentiment score
                if 'POSITIVE' in bert_result['label'].upper():
                    bert_score = bert_result['score']
                elif 'NEGATIVE' in bert_result['label'].upper():
                    bert_score = -bert_result['score']
                else:
                    bert_score = 0
                scores.append(bert_score)
                confidences.append(bert_result['score'])
            except Exception as e:
                print(f"BERT error: {e}")
        
        # Weighted ensemble
        if not scores:
            return {'compound': 0, 'confidence': 0, 'method': 'failed'}
        
        weights = np.array(confidences)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
        
        ensemble_score = np.average(scores, weights=weights)
        ensemble_confidence = np.mean(confidences)
        
        return {
            'compound': float(ensemble_score),
            'confidence': float(ensemble_confidence),
            'method': 'ensemble',
            'individual_scores': {
                'finbert': scores[0] if len(scores) > 0 else None,
                'vader': scores[1] if len(scores) > 1 else None,
                'lexicon': scores[2] if len(scores) > 2 else None,
                'bert': scores[3] if len(scores) > 3 else None
            }
        }
    
    def _finbert_sentiment(self, text):
        """FinBERT-only sentiment analysis"""
        if 'finbert' not in self.models:
            return {'compound': 0, 'confidence': 0, 'method': 'unavailable'}
        
        try:
            result = self.models['finbert'](text)[0]
            score = result['score'] if result['label'] == 'positive' else -result['score']
            if result['label'] == 'neutral':
                score = 0
            
            return {
                'compound': float(score),
                'confidence': float(result['score']),
                'method': 'finbert',
                'label': result['label']
            }
        except Exception as e:
            return {'compound': 0, 'confidence': 0, 'method': 'error', 'error': str(e)}
    
    def _vader_sentiment(self, text):
        """VADER-only sentiment analysis"""
        result = self.models['vader'].polarity_scores(text)
        return {
            'compound': float(result['compound']),
            'confidence': float(abs(result['compound'])),
            'method': 'vader',
            'breakdown': result
        }
    
    def _lexicon_sentiment(self, text):
        """Custom lexicon-only sentiment analysis"""
        score = self.lexicon_based_sentiment(text)
        normalized_score = np.tanh(score)
        
        return {
            'compound': float(normalized_score),
            'confidence': float(min(abs(score), 1.0)),
            'method': 'lexicon',
            'raw_score': float(score)
        }
    
    def batch_analyze(self, texts, method='ensemble'):
        """Analyze sentiment for multiple texts efficiently"""
        results = []
        
        for text in texts:
            result = self.analyze_sentiment(text, method)
            results.append(result)
        
        return results
    
    def analyze_news_impact(self, news_text, company_name=None):
        """
        Analyze news sentiment with additional context about market impact
        """
        basic_sentiment = self.analyze_sentiment(news_text)
        entities = self.extract_financial_entities(news_text)
        
        # Calculate impact score based on content
        impact_multipliers = {
            'earnings': 1.5,
            'merger': 1.3,
            'acquisition': 1.3,
            'lawsuit': 1.2,
            'fraud': 1.8,
            'bankruptcy': 2.0,
            'dividend': 1.1,
            'split': 1.1
        }
        
        impact_score = 1.0
        text_lower = news_text.lower()
        
        for term, multiplier in impact_multipliers.items():
            if term in text_lower:
                impact_score *= multiplier
        
        # Adjust for company mentions
        if company_name and company_name.lower() in text_lower:
            impact_score *= 1.2
        
        return {
            'sentiment': basic_sentiment['compound'],
            'confidence': basic_sentiment['confidence'],
            'impact_score': min(impact_score, 3.0),  # Cap at 3x
            'market_impact': basic_sentiment['compound'] * impact_score,
            'entities': entities,
            'method': basic_sentiment['method']
        }

# Usage Example and Testing
def test_advanced_sentiment_analyzer():
    """Test the advanced sentiment analyzer with sample financial news"""
    
    # Initialize analyzer
    analyzer = AdvancedFinancialSentimentAnalyzer()
    
    # Test cases
    test_texts = [
        "Infosys reported strong Q3 results with revenue growth of 15% and beat analyst expectations",
        "Reliance Industries faces regulatory scrutiny over its recent acquisition deal",
        "Tata Motors stock plunges 8% after disappointing quarterly earnings miss",
        "HDFC Bank announces 12% dividend increase, shares surge in early trading",
        "Nifty 50 index hits record high as FII inflows continue to support market sentiment",
        "Adani Group stocks crash amid fraud allegations and regulatory investigation"
    ]
    
    print("=== Advanced Financial Sentiment Analysis Results ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: {text}")
        
        # Analyze with ensemble method
        result = analyzer.analyze_news_impact(text)
        
        print(f"Sentiment Score: {result['sentiment']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Market Impact: {result['market_impact']:.3f}")
        print(f"Impact Multiplier: {result['impact_score']:.3f}")
        print(f"Entities Found: {result['entities']}")
        print("-" * 80)
    
    return analyzer

# Initialize and test
if __name__ == "__main__":
    # Test the analyzer
    analyzer = test_advanced_sentiment_analyzer()
    
    # Example of batch processing
    sample_news = [
        "RBI keeps repo rate unchanged at 6.5%, maintains accommodative stance",
        "Wipro wins $1.5 billion deal from major European client",
        "Yes Bank reports net loss of Rs 600 crore in Q2 FY25"
    ]
    
    print("\n=== Batch Analysis Results ===")
    batch_results = analyzer.batch_analyze(sample_news)
    
    for text, result in zip(sample_news, batch_results):
        print(f"News: {text[:60]}...")
        print(f"Sentiment: {result['compound']:.3f} (Confidence: {result['confidence']:.3f})")
        print()
