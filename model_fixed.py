import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# Initialize VADER
nltk.download('vader_lexicon', quiet=True)
vader = SentimentIntensityAnalyzer()

# Sample training data (in production would use larger dataset)
train_texts = [
    "This is absolutely wonderful!",
    "I hate this terrible product.",
    "The service was okay, nothing special.",
    "Best experience ever!",
    "Worst decision of my life."
]
train_labels = np.array([1, 0, 0, 1, 0])  # Binary labels (1=positive, 0=negative)

# Create and train Naive Bayes model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(train_texts, train_labels)

def analyze_sentiment(text):
    """Analyze text using both VADER and trained model"""
    # VADER analysis (-1 to 1)
    vader_score = vader.polarity_scores(text)['compound']
    
    # Naive Bayes analysis (0 to 1)
    nb_score = model.predict_proba([text])[0][1]  # Probability of positive class
    
    return vader_score, nb_score