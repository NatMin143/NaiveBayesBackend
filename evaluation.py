# This code are for showing the evaluation of our model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib

# Load the first dataset (CSV)
# This data set comes from https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset
data1 = pd.read_csv('./train.csv', encoding='ISO-8859-1')

# The data here are those in the column 'selected-text' from the datasets
data1_texts = data1['selected_text']

# The data here are those in the column 'label' from the datasets (negative, neutral, or positive) 
data1_labels = data1['sentiment']

# Replace NaN values with an empty string or a placeholder
data1_texts = pd.Series(data1_texts).fillna('')

# Convert text to numerical features using CountVectorizer and do not include stop_words like 'the', 'a', etc.
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data1_texts)

# This splits the data into training and testing sets (80% train, 20% test)
# For this code, we will split the data set into 80% for training and 20% for testing so that we will know how well our model performs on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, data1_labels, test_size=0.2, random_state=42)

# Train the Naive Bayes model
# Explain more
model = MultinomialNB()
model.fit(X_train, y_train)

# # Save the trained model and vectorizer
# joblib.dump(model, 'sentiment_model2.pkl')  # Save the model
# joblib.dump(vectorizer, 'vectorizer2.pkl')  # Save the vectorizer

# Predict the sentiment on the test set
# so the model predicts the sentiment of the test set which are the texts we didn't use to train the model (20% from the data sets)
y_pred = model.predict(X_test)


# This part of the code here is for evaluation of the model


# ------------- Calculate accuracy -------------------
# so from the results of y_pred, it calculate the accuracy of the model to the real labels (y_test)
# For example, in the test set, it has this data "Good": "positive", "Nice but not that good": "neutral", "Bad": "negative"
# It will then use the words "Good", "Nice but not good", "Bad" and predict it sentiments using the model
# So after that , it will compare the results of the model with the real labels and calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Generate precision, recall, and F1-score
# It will result to Precision,  Recall, F1-score, and support

# Precision is how well the model predict, so if the model predicts 100 comments as positive, but 90 of them is actually positive then the precision is 90%
# Recall is how well the model can detect, so if the model can detect 10 comments as positive from the 100 comments, but 2 of them only is actually positive then the recall is 20%
# F1-score is a balance between precision and recall, useful when you want to treat both errors equally.
# Support is the actual comments there were in each category(negative, neutral, positive)
# There are also other datas generated
report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{report}")

# ------------------- Generate confusion matrix ----------------------
# Confusion matrix is a table that is often used to describe the performance of a classification model
# It is a square table where the rows represent the actual classes and the columns represent the predicted classes
# For more info, just search more about this kay medyo lisod e explain diri 
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Perform cross-validation and print the scores
cv_scores = cross_val_score(model, X, data1_labels, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")
