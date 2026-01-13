// Global state
let currentTab = 'text';

// DOM Elements
const textInput = document.getElementById('text-input');
const urlInput = document.getElementById('url-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const errorMessage = document.getElementById('error-message');
const charCounter = document.getElementById('char-counter');

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        switchTab(tab);
    });
});

function switchTab(tab) {
    currentTab = tab;
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tab}-tab`).classList.add('active');
    
    // Hide results and errors
    hideResults();
    hideError();
}

// Character counter
textInput.addEventListener('input', () => {
    const count = textInput.value.length;
    charCounter.textContent = count;
});

// Clear button
clearBtn.addEventListener('click', () => {
    textInput.value = '';
    urlInput.value = '';
    charCounter.textContent = '0';
    hideResults();
    hideError();
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    const content = currentTab === 'text' ? textInput.value : urlInput.value;
    
    if (!content.trim()) {
        showError('Please enter some content to analyze');
        return;
    }
    
    if (currentTab === 'text' && content.trim().length < 10) {
        showError('Text is too short. Please enter at least 10 characters.');
        return;
    }
    
    await analyzeContent(content, currentTab);
});

// Analyze another button
document.getElementById('analyze-another-btn').addEventListener('click', () => {
    hideResults();
    clearBtn.click();
});

// Main analysis function
async function analyzeContent(content, type) {
    showLoading();
    hideError();
    hideResults();
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                content: content,
                type: type
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    } catch (error) {
        showError('An error occurred while analyzing. Please try again.');
        console.error('Error:', error);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(data) {
    const predictionCard = document.getElementById('prediction-card');
    const predictionIcon = document.getElementById('prediction-icon');
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceValue = document.getElementById('confidence-value');
    const reliabilityText = document.getElementById('reliability-text');
    const reliabilityBadge = document.getElementById('reliability-badge');
    
    // Update prediction
    predictionLabel.textContent = data.prediction;
    confidenceValue.textContent = `${data.confidence}%`;
    reliabilityText.textContent = `${data.reliability} Reliability`;
    
    // Update card styling
    predictionCard.className = 'prediction-card';
    if (data.prediction === 'FAKE') {
        predictionCard.classList.add('fake');
        predictionIcon.innerHTML = '<i class="fas fa-times-circle"></i>';
    } else {
        predictionCard.classList.add('real');
        predictionIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
    }
    
    // Update probability bars
    document.getElementById('fake-prob').textContent = `${data.probabilities.fake}%`;
    document.getElementById('real-prob').textContent = `${data.probabilities.real}%`;
    document.getElementById('fake-bar').style.width = `${data.probabilities.fake}%`;
    document.getElementById('real-bar').style.width = `${data.probabilities.real}%`;
    
    // Update features
    displayFeatures(data.features);
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display features
function displayFeatures(features) {
    const featuresGrid = document.getElementById('features-grid');
    featuresGrid.innerHTML = '';
    
    const featureConfig = {
        'char_count': { label: 'Characters', icon: 'fa-text-width' },
        'word_count': { label: 'Words', icon: 'fa-font' },
        'sentence_count': { label: 'Sentences', icon: 'fa-paragraph' },
        'exclamation_count': { label: 'Exclamations', icon: 'fa-exclamation' },
        'question_count': { label: 'Questions', icon: 'fa-question' },
        'sensational_count': { label: 'Sensational Words', icon: 'fa-star' },
        'polarity': { label: 'Sentiment Polarity', icon: 'fa-smile' },
        'subjectivity': { label: 'Subjectivity', icon: 'fa-brain' },
        'capital_ratio': { label: 'Capital Ratio', icon: 'fa-text-height' },
        'unique_word_ratio': { label: 'Unique Words', icon: 'fa-fingerprint' }
    };
    
    for (const [key, value] of Object.entries(features)) {
        if (featureConfig[key]) {
            const config = featureConfig[key];
            let displayValue = value;
            
            // Format values
            if (key.includes('ratio') || key.includes('polarity') || key.includes('subjectivity')) {
                displayValue = (value * 100).toFixed(1) + '%';
            } else if (typeof value === 'number') {
                displayValue = Math.round(value);
            }
            
            const featureItem = document.createElement('div');
            featureItem.className = 'feature-item';
            featureItem.innerHTML = `
                <div class="feature-label">
                    <i class="fas ${config.icon}"></i>
                    ${config.label}
                </div>
                <div class="feature-value">${displayValue}</div>
            `;
            featuresGrid.appendChild(featureItem);
        }
    }
}

// Utility functions
function showLoading() {
    loading.style.display = 'block';
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loading.style.display = 'none';
    analyzeBtn.disabled = false;
}

function showError(message) {
    error
