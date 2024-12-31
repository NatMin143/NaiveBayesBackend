# This file is to simplify how to create a naive-bayes model and train/test data sets
# It only uses simple/small number of datasets

# Import necessary libraries
# Explain what is the purpose of each library
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data sets
train_texts = [
    "This product is very good", "This is awesome", 
    "This is okay but not great", "I don't have opinion on this yet", 
    "I hate this product", "This is the worst"
]

# These are the labels associated to the texts above
# This is like
# "This product is very good" is positive, "This is awesome" is positive, 
# "This is okay but not great is neutral" and so on
train_labels = ["positive", "positive", "neutral", "neutral", "negative", "negative"]

# Convert text to numerical features using CountVectorizer and do not include stop_words like 'the', 'a', etc.
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_texts)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, train_labels)

# Predict sentiment on some test data 
test_texts = [
    "This is good", 
    "The product arrived on time, but I haven’t used it enough to form a strong opinion yet.", 
    "This is the more than worst"
]

# Convert test data to numerical features
test_X = vectorizer.transform(test_texts)

# Predict sentiment of the test data
predictions = model.predict(test_X)

print("Predictions:", predictions)
