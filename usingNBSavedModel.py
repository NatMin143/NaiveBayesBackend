# This file is a simple way on how to use the trained model from the nb_model_trainer.py
import joblib

# Load the saved model and vectorizer
model = joblib.load('sentiment_model2.pkl')
vectorizer = joblib.load('vectorizer2.pkl')

# Predict sentiment on new test data
test_texts = ["Wow. I love this!"]
test_X = vectorizer.transform(test_texts)
predictions = model.predict(test_X)

print("Predictions:", predictions[0])
