# spam_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------- Preprocessing Function --------------------
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = re.sub('[^a-zA-Z]', ' ', text)     # Remove special characters
    text = text.lower().split()               # Lowercase and split
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
    if 'not' in stop_words:
        stop_words.remove('not')
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# -------------------- Load & Clean Data --------------------
def load_and_prepare_data(filepath):
    dataset = pd.read_csv(filepath, encoding='ISO-8859-1')
    corpus = [preprocess_text(message) for message in dataset['Message_body']]
    return corpus, dataset.iloc[:, -1].values

# -------------------- Train Model --------------------
def train_model(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

# -------------------- Predict Single Message --------------------
def predict_new_message(classifier, vectorizer, message):
    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed]).toarray()
    prediction = classifier.predict(vectorized)
    return prediction

# -------------------- Main Pipeline --------------------
def main():
    nltk.download('stopwords')

    # Load & preprocess data
    filepath = r'C:\Users\abhin\Downloads\Logistic_Regression\FinalFolder\Dataset\SMS_train.csv'
    corpus, y = load_and_prepare_data(filepath)

    # Convert text to feature vectors
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Train model
    classifier = train_model(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)
    print("Predictions vs Actual:\n", np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

    # Evaluation
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print(f"\nAccuracy Score: {accuracy:.4f}")

    # Test on a custom input
    custom_msg = "Be alert of spams!!"
    custom_pred = predict_new_message(classifier, cv, custom_msg)
    print(f"\nCustom Message Prediction: {custom_pred[0]}")  # 1 or 0

if __name__ == "__main__":
    main()