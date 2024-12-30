from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

# Load the saved model and vectorizer
model = joblib.load('sentiment_model2.pkl')
vectorizer = joblib.load('vectorizer2.pkl')

app = Flask(__name__)

CORS(app)

# Root route
@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Parse the incoming JSON payload
        text = data.get('text')

        print("HELLOW WORLD")

        if not text:
            return jsonify({'error': 'No text provided for prediction.'}), 400

        # Process the input text
        test_X = vectorizer.transform([text])
        prediction = model.predict(test_X)

        print("Prediction: ", prediction)

        # Convert numpy int32 to native Python int
        prediction_value = (prediction[0])

        # Return the prediction as a response
        return jsonify({'prediction': prediction_value})  # Now it's a serializable int
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
