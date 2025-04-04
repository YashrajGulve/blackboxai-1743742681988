# Sentiment Analysis Web Application

## Project Overview
A Flask web application that performs sentiment analysis using:
- VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK
- Custom-trained Naive Bayes classifier

## Features
- Modern responsive interface with Tailwind CSS
- Dual-model analysis (VADER + Naive Bayes)
- Interactive results visualization
- Easy deployment to Render

## Installation
```bash
# Clone repository
git clone [repository-url]
cd sentiment-analysis-app

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage
```bash
# Run development server
python app.py

# Access application at:
http://localhost:5000
```

## Deployment to Render
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set configuration:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Deploy!

## Project Structure
```
.
├── app.py                # Flask application
├── model.py              # ML models and analysis
├── requirements.txt      # Python dependencies
├── Procfile              # Deployment configuration
└── templates/
    ├── index.html        # Main interface
    └── analysis_results.html # Results page
```

## License
MIT