import streamlit as st
import pickle
import numpy as np
import os

# Define the path to load the model
directory = r'D:\\Internship 1st\\Assignment 1\\Sales_forecasting'
filename = os.path.join(directory, 'sales_prediction_model.pkl')

# Load the model from disk
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Set page config for dark theme
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': 'https://www.example.com/bug',
        'About': 'This is a simple app to predict sales based on advertising spend.'
    }
)

# Set custom CSS for dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Sales Prediction App")

st.markdown("#### Enter the advertising spend for each medium below:")

# Define the feature columns
feature_cols = ['TV', 'radio', 'newspaper']

# Create inputs for each feature
inputs = []
for feature in feature_cols:
    value = st.number_input(f'{feature} spend', min_value=0.0, value=0.0, step=0.1, format="%.2f")
    inputs.append(value)

# Convert inputs to numpy array for prediction
inputs = np.array(inputs).reshape(1, -1)

if st.button('Predict'):
    prediction = model.predict(inputs)
    st.write(f'### Predicted Sales: {prediction[0]:.2f}')
