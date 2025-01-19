import joblib
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer from the saved folder
model = joblib.load('spam_classifier.pkl')
cv = joblib.load('tfidf_vectorizer.pkl')

# Streamlit UI components
st.title("SMS Spam Classifier")
st.write("This app predicts whether your SMS is spam or not.")

# Text input for prediction
input_message = st.text_area("Enter the SMS message:")

# "Classify" button
if st.button("Classify"):
    if input_message:
        # Initialize Lemmatizer
        lem = WordNetLemmatizer()

        # Preprocess the input message
        review = re.sub('[^a-zA-Z]', ' ', input_message)
        review = review.lower()
        review = review.split()
        review = [lem.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        
        # Transform the input message using TF-IDF
        input_vector = cv.transform([review]).toarray()
        
        # Predict if the message is spam or not
        prediction = model.predict(input_vector)
        
        # Display prediction result
        if prediction == 0:
            st.write("This is a **ham** message (not spam).")
        else:
            st.write("This is a **spam** message.")
    else:
        st.write("Please enter a message to classify.")
