import streamlit as st
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Load your model and the CountVectorizer
with open('Sentiment_Analysis_Movie_review_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('count_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Set the theme to dark
st.set_page_config(page_title="Movie Review Predictor", page_icon="🎬", layout="centered", initial_sidebar_state="auto")

# Custom CSS for the dark theme
st.markdown("""
    <style>
    body {
        color: #fff;
        background-color: #000;
    }
    .stTextInput > div > div > input {
        color: #fff;
        background-color: #333;
    }
    .stButton > button {
        color: #fff;
        background-color: #444;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the app
st.title("🎬 Movie Review Predictor")

# Function to clean HTML tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Function to convert text to lowercase
def convert_lower(text):
    return text.lower()

# Function to remove special characters
def remove_special(text):
    x = ''
    for i in text:
        if i.isalnum():
            x = x + i
        else:
            x = x + ' '
    return x

# Function to remove stopwords
def remove_stopwords(text):
    x = []
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)
    y = x[:]
    x.clear()
    return y

# Function to apply stemming
ps = PorterStemmer()
def stem_words(text):
    y = []
    for i in text:
        y.append(ps.stem(i))
    z = y[:]
    y.clear()
    return z

# Function to join the words back into a single string
def join_back(list_input):
    return " ".join(list_input)

# Input from the user
user_input = st.text_input("Enter your movie review:")

# Predict button
if st.button("Predict"):
    if user_input:
        # Preprocess the input
        preprocessed_input = clean_html(user_input)
        preprocessed_input = convert_lower(preprocessed_input)
        preprocessed_input = remove_special(preprocessed_input)
        preprocessed_input = remove_stopwords(preprocessed_input)
        preprocessed_input = stem_words(preprocessed_input)
        preprocessed_input = join_back(preprocessed_input)
        
        # Transform the input using the CountVectorizer
        transformed_input = vectorizer.transform([preprocessed_input])
        
        # Make a prediction
        prediction = model.predict(transformed_input)
        st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
    else:
        st.write("Please enter a review to predict.")

# Disclaimer
st.markdown("""
    <hr>
    <p style='text-align: center;'>This application is powered by a machine learning model.</p>
    """, unsafe_allow_html=True)
