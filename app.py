import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from utils.detector import AdvancedFakeNewsDetector
from utils.scraper import ArticleScraper

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize detector and scraper
detector = AdvancedFakeNewsDetector()
scraper = ArticleScraper()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze news article"""
    data = request.get_json()
    
    input_type = data.get('type', 'text')
    
    if input_type == 'url':
        url = data.get('content', '')
        article_data = scraper.scrape_article(url)
        
        if 'error' in article_data:
            return jsonify(article_data)
        
        text = article_data.get('text', '')
        title = article_data.get('title', '')
        full_text = f"{title} {text}"
    else:
        full_text = data.get('content', '')
    
    if not full_text or len(full_text.strip()) < 10:
        return jsonify({'error': 'Text too short or empty'})
    
    # Predict
    result = detector.predict(full_text)
    
    return jsonify(result)

@app.route('/train', methods=['POST'])
def train():
    """Train the model"""
    try:
        accuracy = detector.train_model()
        return jsonify({
            'success': True,
            'accuracy': round(accuracy * 100, 2),
            'message': 'Model trained successfully!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': detector.is_trained
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Advanced Fake News Detection System")
    print("=" * 60)
    
    # Try to load existing model
    if not detector.load_model():
        print("\nNo trained model found.")
        print("Training new model...")
        detector.train_model()
    
    print("\nStarting Flask server...")
    print("Visit: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000)
