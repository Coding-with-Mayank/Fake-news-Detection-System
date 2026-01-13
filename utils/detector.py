import os
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from langdetect import detect
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AdvancedFakeNewsDetector:
    """Enhanced Fake News Detection System with Advanced Features"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Advanced text preprocessing"""
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def extract_advanced_features(self, text):
        """Extract comprehensive linguistic features"""
        features = {}
        
        # Basic length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['period_count'] = text.count('.')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # Sensational/clickbait indicators
        sensational_words = ['shocking', 'unbelievable', 'amazing', 'secret', 'miracle', 
                            'exposed', 'breakthrough', 'stunning', 'incredible', 'urgent']
        features['sensational_count'] = sum(1 for word in sensational_words if word in text.lower())
        
        # Sentiment analysis using TextBlob
        try:
            blob = TextBlob(text)
            features['polarity'] = blob.sentiment.polarity
            features['subjectivity'] = blob.sentiment.subjectivity
        except:
            features['polarity'] = 0
            features['subjectivity'] = 0
        
        # Language detection
        try:
            features['is_english'] = 1 if detect(text) == 'en' else 0
        except:
            features['is_english'] = 1
        
        # Readability metrics
        words = text.split()
        if len(words) > 0:
            features['unique_word_ratio'] = len(set(words)) / len(words)
        else:
            features['unique_word_ratio'] = 0
        
        # Quote presence
        features['has_quotes'] = 1 if '"' in text or "'" in text else 0
        
        # ALL CAPS words (shouting)
        features['caps_word_count'] = sum(1 for word in words if word.isupper() and len(word) > 2)
        
        return features
    
    def train_model(self, fake_csv='data/Fake.csv', true_csv='data/True.csv'):
        """Train enhanced model with additional features"""
        print("Loading datasets...")
        
        if not os.path.exists(fake_csv) or not os.path.exists(true_csv):
            print("Warning: Training data not found. Creating sample data...")
            self.create_sample_data()
        
        df_fake = pd.read_csv(fake_csv)
        df_true = pd.read_csv(true_csv)
        
        # Add labels
        df_fake['label'] = 0  # Fake
        df_true['label'] = 1  # Real
        
        # Combine datasets
        df = pd.concat([df_fake, df_true], ignore_index=True)
        
        # Handle text column
        if 'text' in df.columns:
            df = df[['text', 'label']]
        elif 'title' in df.columns:
            df['text'] = df['title'].fillna('') + ' ' + df.get('text', '').fillna('')
            df = df[['text', 'label']]
        
        print(f"Total samples: {len(df)}")
        print(f"Fake news: {sum(df['label'] == 0)}, Real news: {sum(df['label'] == 1)}")
        
        # Clean text
        print("Preprocessing text...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Extract additional features
        print("Extracting advanced features...")
        additional_features = df['text'].apply(self.extract_advanced_features)
        feature_df = pd.DataFrame(additional_features.tolist())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df[['cleaned_text']], df['label'],
            test_size=0.2, random_state=42, stratify=df['label']
        )
        
        X_train_features, X_test_features = train_test_split(
            feature_df, test_size=0.2, random_state=42, stratify=df['label']
        )
        
        # TF-IDF Vectorization
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train['cleaned_text'])
        X_test_tfidf = self.vectorizer.transform(X_test['cleaned_text'])
        
        # Scale additional features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_test_scaled = self.scaler.transform(X_test_features)
        
        # Combine TF-IDF and additional features
        from scipy.sparse import hstack
        X_train_combined = hstack([X_train_tfidf, X_train_scaled])
        X_test_combined = hstack([X_test_tfidf, X_test_scaled])
        
        # Train Random Forest model
        print("Training Enhanced Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.model.fit(X_train_combined, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.is_trained = True
        self.save_model()
        
        return accuracy
    
    def predict(self, text):
        """Predict with enhanced features"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Extract additional features
        additional_features = self.extract_advanced_features(text)
        feature_array = np.array([list(additional_features.values())])
        feature_scaled = self.scaler.transform(feature_array)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([cleaned])
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([text_tfidf, feature_scaled])
        
        # Predict
        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0]
        
        confidence = probability[prediction] * 100
        
        # Determine reliability
        if confidence >= 85:
            reliability = "High"
        elif confidence >= 70:
            reliability = "Medium"
        else:
            reliability = "Low"
        
        result = {
            'prediction': 'REAL' if prediction == 1 else 'FAKE',
            'confidence': round(confidence, 2),
            'reliability': reliability,
            'label': int(prediction),
            'probabilities': {
                'fake': round(probability[0] * 100, 2),
                'real': round(probability[1] * 100, 2)
            },
            'features': additional_features
        }
        
        return result
    
    def create_sample_data(self):
        """Create sample training data"""
        os.makedirs('data', exist_ok=True)
        
        fake_samples = [
            "SHOCKING: Scientists discover miracle cure that doctors don't want you to know!",
            "BREAKING: Aliens confirmed by government officials in secret meeting!",
            "You won't believe what this celebrity said about politics!",
            "This one weird trick will make you rich overnight!",
            "Government hiding the truth about this dangerous conspiracy!"
        ] * 100
        
        real_samples = [
            "According to recent studies published in Nature, researchers have made progress in understanding climate change.",
            "The university announced new findings in medical research conducted over three years.",
            "Economic indicators suggest steady growth in the manufacturing sector, experts report.",
            "Scientists at MIT have developed a new approach to renewable energy storage.",
            "Government officials announced policy changes following extensive public consultation."
        ] * 100
        
        pd.DataFrame({'text': fake_samples}).to_csv('data/Fake.csv', index=False)
        pd.DataFrame({'text': real_samples}).to_csv('data/True.csv', index=False)
        print("Sample data created successfully!")
    
    def save_model(self, model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl', scaler_path='models/scaler.pkl'):
        """Save trained model, vectorizer, and scaler"""
        os.makedirs('models', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl', scaler_path='models/scaler.pkl'):
        """Load trained model, vectorizer, and scaler"""
        if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            print("Model loaded successfully!")
            return True
        return False
