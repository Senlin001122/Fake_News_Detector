from flask import Flask, request, render_template
import joblib
import os
import requests
from bs4 import BeautifulSoup

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.joblib')
vectorizer_path = os.path.join(current_dir, 'vectorizer.joblib')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = Flask(__name__)

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except Exception as e:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    url = request.form.get('url', '')

    if url:
        text = extract_text_from_url(url)
        if not text:
            return render_template('results.html', prediction=None, confidence=None, error="Couldn't fetch the article from the provided URL.")

    if not text:
        return render_template('results.html', prediction=None, confidence=None, error="No text or URL provided.")

    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf).max()

    # Format confidence as percentage rounded to two decimal places
    formatted_confidence = f"{confidence * 100:.2f}%"

    return render_template('results.html', prediction=prediction, confidence=formatted_confidence, error=None)

if __name__ == '__main__':
    app.run(debug=True)
