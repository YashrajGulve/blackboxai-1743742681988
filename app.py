from flask import Flask, render_template, request, jsonify
from model_fixed import analyze_sentiment
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.form['text']
        if not text or len(text.strip()) < 10:
            return jsonify({'error': 'Please enter at least 10 characters'}), 400
            
        logger.info(f"Analyzing text: {text[:50]}...")
        vader_score, nb_score = analyze_sentiment(text)
        
        return render_template('results.html', 
                             vader_score=vader_score,
                             nb_score=nb_score)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)