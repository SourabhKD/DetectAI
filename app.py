from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
max_length = 100

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    global model, tokenizer
    
    try:
        # Load the model
        model = tf.keras.models.load_model('ai_detection_model.keras')
        print("Model loaded successfully!")
        
        # Load the tokenizer
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
        print("Tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return False
    
    return True

def predict_ai_generated(text):
    """Predict if text is AI-generated or human-written"""
    if not model or not tokenizer:
        return None, "Model not loaded"
    
    try:
        # Preprocess the text
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        confidence = float(prediction)
        
        # Determine result
        if prediction > 0.5:
            result = "AI-generated"
            confidence_percent = confidence * 100
        else:
            result = "Human-written"
            confidence_percent = (1 - confidence) * 100
            
        return result, confidence_percent
    
    except Exception as e:
        return None, f"Prediction error: {e}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Please provide some text to analyze'
            })
        
        if len(text) < 10:
            return jsonify({
                'success': False,
                'error': 'Text is too short for reliable analysis (minimum 10 characters)'
            })
        
        result, confidence = predict_ai_generated(text)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': confidence  # confidence contains error message in this case
            })
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'text_length': len(text),
            'word_count': len(text.split())
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    # Load model and tokenizer on startup
    if load_model_and_tokenizer():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model and tokenizer. Please ensure the files exist in the current directory.")
        print("Required files: 'ai_detection_model.keras' and 'tokenizer.json'")