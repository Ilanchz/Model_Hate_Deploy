from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_classifier = pickle.load(file)

# Load the vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the request
        app.logger.info('Received request: %s', request.json)

        # Get text input from the user
        data = request.get_json(force=True)
        text = data['text']

        # Preprocess the text
        text_piece = [text]
        X_text_piece = vectorizer.transform(text_piece)

        # Predict class label
        prediction = logistic_classifier.predict(X_text_piece)[0]

        # Log the response
        app.logger.info('Sending response: %s', {'prediction': int(prediction)})

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        # Log any errors
        app.logger.error('Error: %s', str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)

