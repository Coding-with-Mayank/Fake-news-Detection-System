# Fake News Detection API Documentation

## Overview
This API provides endpoints for analyzing news articles to detect fake news using machine learning.

## Base URL
```
http://localhost:5000
```

## Endpoints

### 1. Health Check
Check if the API is running and if the model is trained.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_trained": true
}
```

---

### 2. Analyze Text/URL
Analyze news content to determine if it's fake or real.

**Endpoint:** `POST /analyze`

**Request Body:**
```json
{
  "type": "text",
  "content": "Your news article text here..."
}
```

OR
```json
{
  "type": "url",
  "content": "https://example.com/news-article"
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 94.7,
  "reliability": "High",
  "label": 0,
  "probabilities": {
    "fake": 94.7,
    "real": 5.3
  },
  "features": {
    "char_count": 1234,
    "word_count": 200,
    "sentence_count": 15,
    "exclamation_count": 5,
    "question_count": 2,
    "sensational_count": 3,
    "polarity": 0.15,
    "subjectivity": 0.62,
    "capital_ratio": 0.08,
    "unique_word_ratio": 0.72
  }
}
```

**Status Codes:**
- `200 OK` - Successful analysis
- `400 Bad Request` - Invalid input
- `500 Internal Server Error` - Server error

---

### 3. Train Model
Trigger model training (requires training data files).

**Endpoint:** `POST /train`

**Response:**
```json
{
  "success": true,
  "accuracy": 92.3,
  "message": "Model trained successfully!"
}
```

---

## Error Responses

All endpoints may return error responses in this format:
```json
{
  "error": "Error message description"
}
```

---

## Rate Limiting
Currently no rate limiting is implemented. For production use, consider implementing rate limiting.

---

## Example Usage

### Using cURL
```bash
# Analyze text
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text",
    "content": "SHOCKING: Scientists discover miracle cure!"
  }'

# Analyze URL
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "type": "url",
    "content": "https://example.com/article"
  }'

# Health check
curl http://localhost:5000/health
```

### Using Python
```python
import requests

# Analyze text
response = requests.post('http://localhost:5000/analyze', json={
    'type': 'text',
    'content': 'Your news article text here...'
})
result = response.json()
print(result)

# Analyze URL
response = requests.post('http://localhost:5000/analyze', json={
    'type': 'url',
    'content': 'https://example.com/article'
})
result = response.json()
print(result)
```

### Using JavaScript
```javascript
// Analyze text
fetch('http://localhost:5000/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    type: 'text',
    content: 'Your news article text here...'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## Features Explained

### Linguistic Features

1. **char_count**: Total number of characters in the text
2. **word_count**: Total number of words
3. **sentence_count**: Number of sentences
4. **exclamation_count**: Number of exclamation marks (!)
5. **question_count**: Number of question marks (?)
6. **sensational_count**: Count of sensational/clickbait words
7. **polarity**: Sentiment polarity (-1 to 1, negative to positive)
8. **subjectivity**: Text subjectivity (0 to 1, objective to subjective)
9. **capital_ratio**: Ratio of capital letters to total characters
10. **unique_word_ratio**: Ratio of unique words to total words

### Prediction Labels

- **FAKE** (label: 0): The article is likely fake news
- **REAL** (label: 1): The article is likely real news

### Reliability Levels

- **High**: Confidence ≥ 85%
- **Medium**: Confidence ≥ 70%
- **Low**: Confidence < 70%

---

## API Integration with External Services

### Optional: News API Integration

If you have a News API key, you can add it to `.env`:
```env
NEWS_API_KEY=your-newsapi-key
```

This allows fetching articles from News API for verification.

### Optional: Google Fact Check API

Add Google Fact Check API key to `.env`:
```env
GOOGLE_FACT_CHECK_API_KEY=your-google-factcheck-key
```

This enables cross-referencing with Google's Fact Check database.

---

## Best Practices

1. **Input Validation**: Always validate input before sending to the API
2. **Error Handling**: Implement proper error handling for API responses
3. **Timeout**: Set appropriate timeout values for requests
4. **Caching**: Consider caching results for frequently analyzed URLs
5. **Security**: Use HTTPS in production environments

---

## Limitations

1. English language only (for now)
2. Requires minimum 10 characters of text
3. URL scraping may fail for some websites
4. No authentication/authorization (add for production)
5. Model accuracy depends on training data quality

---

## Support

For issues or questions:
- Create an issue on GitHub
- Contact: [Your Email]
