# This file is use to train the model before-hand so that I can use these trained model in other programs

# Importing necessary libraries
# There are libraries added
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# use to save the model and vectorizer
import joblib

# Load the first dataset (CSV)
data1 = pd.read_csv('./train.csv', encoding='ISO-8859-1')
data1_texts = data1['selected_text']
data1_labels = data1['sentiment']


# Replace NaN values with an empty string or a placeholder
data1_texts = pd.Series(data1_texts).fillna('')

# Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(data1_texts)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, data1_labels)

# Save the trained model and vectorizer
joblib.dump(model, 'sentiment_model2.pkl')  # Save the model
joblib.dump(vectorizer, 'vectorizer2.pkl')  # Save the vectorizer
