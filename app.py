from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import pandas as pd
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and vectorizer
print("Loading model and vectorizer...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully!")

# Class labels mapping
CLASS_LABELS = {0: "hate speech", 1: "offensive language", 2: "neither"}

def clean_tweet(tweet):
    """Clean and preprocess tweet text"""
    tweet = str(tweet).lower()
    tweet = re.sub(r'http\S+|https\S+', '', tweet)
    tweet = re.sub(r'@\S+', '', tweet)
    tweet = re.sub(r'#\S+', '', tweet)
    tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

def get_word_analysis(tweet, vectorized_tweet, predicted_class):
    """Get word-level analysis with TF-IDF scores and coefficients"""
    words = tweet.split()
    word_scores = []
    
    coefficients = model.coef_[predicted_class]
    
    for word in words:
        if word in vectorizer.vocabulary_:
            word_index = vectorizer.vocabulary_[word]
            tfidf_score = vectorized_tweet[0, word_index]
            coefficient = coefficients[word_index]
            
            if tfidf_score > 0:  # Only include words with non-zero TF-IDF
                word_scores.append({
                    'word': word,
                    'tfidf': float(tfidf_score),
                    'coefficient': float(coefficient)
                })
    
    # Sort by absolute coefficient value and return top 8
    word_scores.sort(key=lambda x: abs(x['coefficient']), reverse=True)
    return word_scores[:8]

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Tweet Behavior Analysis API is running'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_tweet():
    """Analyze a single tweet"""
    try:
        data = request.get_json()
        
        if not data or 'tweet' not in data:
            return jsonify({'error': 'No tweet provided'}), 400
        
        tweet = data['tweet']
        
        if not tweet or not tweet.strip():
            return jsonify({'error': 'Tweet cannot be empty'}), 400
        
        # Clean and vectorize the tweet
        cleaned_tweet = clean_tweet(tweet)
        tweet_vectorized = vectorizer.transform([cleaned_tweet])
        
        # Predict class and get probability
        predicted_class = model.predict(tweet_vectorized)[0]
        probabilities = model.predict_proba(tweet_vectorized)[0]
        confidence = float(probabilities[predicted_class] * 100)
        
        # Get word-level analysis
        word_scores = get_word_analysis(cleaned_tweet, tweet_vectorized, predicted_class)
        
        # Prepare response
        response = {
            'predictedClass': CLASS_LABELS[predicted_class],
            'confidence': confidence,
            'wordScores': word_scores,
            'probabilities': {
                'hateSpeech': float(probabilities[0] * 100),
                'offensiveLanguage': float(probabilities[1] * 100),
                'neither': float(probabilities[2] * 100)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk-analyze', methods=['POST'])
def bulk_analyze():
    """Analyze multiple tweets from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        if 'tweets' not in df.columns:
            return jsonify({'error': 'CSV must contain a "tweets" column'}), 400
        
        # Clean and predict
        df['cleaned_tweet'] = df['tweets'].apply(clean_tweet)
        tweet_vectorized = vectorizer.transform(df['cleaned_tweet'])
        predicted_classes = model.predict(tweet_vectorized)
        
        # Add predictions to dataframe
        df['predicted_behavior'] = [CLASS_LABELS[c] for c in predicted_classes]
        
        # Calculate statistics
        stats = {
            'total': len(df),
            'hateSpeech': int((predicted_classes == 0).sum()),
            'offensive': int((predicted_classes == 1).sum()),
            'neither': int((predicted_classes == 2).sum())
        }
        
        # Generate output CSV
        output_df = df[['tweets', 'predicted_behavior']]
        output_csv = output_df.to_csv(index=False)
        
        return jsonify({
            'stats': stats,
            'csv': output_csv
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development
    # app.run(debug=True, host='0.0.0.0', port=5000)
