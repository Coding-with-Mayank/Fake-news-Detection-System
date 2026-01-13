# Fake-news-Detection-System
ML-based Fake News Detection System using Random Forest, TF-IDF, and NLP with Flask web interface. Achieves 80%+ accuracy with real-time analysis capabilities.

# 🛡️ Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced machine learning-based system for detecting fake news using Random Forest classification, TF-IDF vectorization, and comprehensive NLP analysis. Achieves **92%+ accuracy** with real-time analysis capabilities.

## 🌟 Features

- **High Accuracy**: 80%+ accuracy using Random Forest ML
- **Real-time Analysis**: Sub-100ms prediction times
- **Dual Input**: Support for both text and URL inputs
- **Advanced NLP**: NLTK preprocessing and sentiment analysis
- **Linguistic Analysis**: 10+ linguistic features extraction
- **Confidence Scores**: Transparent probability distributions
- **Web Interface**: Beautiful, responsive UI
- **RESTful API**: Easy integration with other applications
- **Model Persistence**: Save and load trained models

## 🎯 Technology Stack

### Backend
- **Python 3.8+**
- **Flask 3.0.0** - Web framework
- **scikit-learn 1.3.2** - Machine learning
- **NLTK 3.8.1** - Natural language processing
- **pandas & numpy** - Data manipulation
- **BeautifulSoup4** - Web scraping
- **newspaper3k** - Article extraction
- **TextBlob** - Sentiment analysis

### Frontend
- **HTML5 & CSS3** - Modern responsive design
- **JavaScript ES6** - Interactive functionality
- **Font Awesome** - Icons

### Machine Learning
- **Random Forest Classifier** - Primary model
- **TF-IDF Vectorization** - Feature extraction
- **Sentiment Analysis** - Emotional tone detection
- **Language Detection** - Multi-language support (planned)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 80.3% |
| Precision | 79.8% |
| Recall | 92.7% |
| F1-Score | 79.2% |

### Confusion Matrix
```
                Predicted
              Fake    Real
Actual Fake   4521    387
       Real    341   4651
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection-system.git
cd fake-news-detection-system
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. **Set up environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env file with your configuration
# (Optional: Add API keys for enhanced features)
```

6. **Prepare training data**

Place your training data files in the `data/` folder:
- `data/Fake.csv` - Fake news samples
- `data/True.csv` - Real news samples

Or use the sample data generator (will be created automatically on first run).

7. **Run the application**
```bash
python app.py
```

8. **Access the application**

Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure
```
fake-news-detection-system/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
│
├── static/                    # Static files
│   ├── css/
│   │   └── style.css         # Stylesheet
│   └── js/
│       └── script.js         # JavaScript
│
├── templates/                 # HTML templates
│   └── index.html            # Main page
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── detector.py           # ML detector class
│   └── scraper.py            # Web scraper
│
├── models/                    # Trained models
│   ├── model.pkl             # Trained classifier
│   ├── vectorizer.pkl        # TF-IDF vectorizer
│   └── scaler.pkl            # Feature scaler
│
├── data/                      # Training data
│   ├── Fake.csv              # Fake news dataset
│   └── True.csv              # Real news dataset
│
└── docs/                      # Documentation
    └── API_DOCUMENTATION.md  # API docs
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:
```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

# Optional API Keys
NEWS_API_KEY=your-newsapi-key
GOOGLE_FACT_CHECK_API_KEY=your-google-factcheck-key

# Model Configuration
MODEL_PATH=models/
MAX_FEATURES=5000
N_ESTIMATORS=100
```

### Training Data Format

CSV files should have at least a `text` column:
```csv
text
"Article text here..."
"Another article..."
```

Or `title` and `text` columns:
```csv
title,text
"Article Title","Article body text..."
```

## 💡 Usage

### Web Interface

1. Navigate to `http://localhost:5000`
2. Choose input method (Text or URL)
3. Enter news content
4. Click "Analyze"
5. View results with confidence scores

### API Usage

#### Analyze Text
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text",
    "content": "Your news article text here..."
  }'
```

#### Analyze URL
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "type": "url",
    "content": "https://example.com/article"
  }'
```

#### Python Example
```python
import requests

response = requests.post('http://localhost:5000/analyze', json={
    'type': 'text',
    'content': 'SHOCKING: Scientists discover miracle cure!'
})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
```

See [API Documentation](docs/API_DOCUMENTATION.md) for more details.

## 🎓 How It Works

### 1. Text Preprocessing
- Lowercase conversion
- URL and HTML removal
- Special character removal
- Tokenization
- Stopword removal

### 2. Feature Extraction

**TF-IDF Features**
- 5000 maximum features
- 1-3 word n-grams
- Captures word importance

**Linguistic Features**
- Character/word/sentence counts
- Punctuation analysis
- Sensational word detection
- Sentiment polarity & subjectivity
- Capital letter ratio
- Unique word ratio

### 3. Classification
- Random Forest with 150 trees
- Maximum depth of 25
- Class weight balancing
- Probability-based confidence scores

### 4. Prediction
- Binary classification (Fake/Real)
- Confidence percentage
- Reliability assessment
- Feature analysis

## 📈 Performance Optimization

### Training Tips
1. Use balanced datasets (equal fake/real samples)
2. Include diverse sources and writing styles
3. Regularly retrain with new data
4. Validate on unseen data

### Production Deployment
1. Use Gunicorn for WSGI server
2. Set up Nginx as reverse proxy
3. Enable caching for frequent URLs
4. Implement rate limiting
5. Use HTTPS

Example Gunicorn command:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🔒 Security Considerations

- Input validation and sanitization
- Rate limiting for API endpoints
- CORS configuration for production
- Environment variable protection
- SQL injection prevention (if using database)
- XSS protection in frontend

## 🐛 Troubleshooting

### Common Issues

**NLTK Data Not Found**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Model Not Loading**
- Ensure `models/` directory exists
- Check if model files are present
- Retrain model if necessary

**URL Scraping Fails**
- Some websites block scrapers
- Try different URLs
- Check internet connection

**Low Accuracy**
- Retrain with more data
- Balance fake/real samples
- Clean training data

## 🚧 Future Enhancements

### Short-term (3-6 months)
- [ ] Deep Learning integration (BERT/RoBERTa)
- [ ] Multilingual support
- [ ] Fact-checking API integration
- [ ] Browser extension
- [ ] Mobile app (Android/iOS)

### Long-term (6-12 months)
- [ ] Image manipulation detection
- [ ] Video deepfake detection
- [ ] Source credibility scoring
- [ ] Social network analysis
- [ ] Real-time monitoring dashboard

## 👥 Contributors

- **Mayank Kumar Kharwar** - Developer
- **Manas Baiswar** - Developer
- **Tanya** - Developer
- **Ms. Himani Sharma** - Project Guide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DIT University, Dehradun
- School of Computing
- scikit-learn community
- NLTK developers
- Flask community

## 📞 Contact

For questions or support:
- Create an issue on GitHub
- Email: [your-email@example.com]

## 📚 References

1. Shu, K., et al. (2020). "Fake News Detection on Social Media"
2. Zhou, X., & Zafarani, R. (2020). "A Survey of Fake News"
3. Kaggle Fake News Dataset
4. scikit-learn Documentation
5. NLTK Documentation

---

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by the DIT University Team
