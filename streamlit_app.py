# streamlit_app.py

import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("🌼 Iris Flower Classifier")
st.write(
    """
    This simple app uses a trained Random Forest to classify an iris flower 
    based on its sepal and petal dimensions.
    """
)

# Input sliders
sepal_length = st.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal width (cm)', 0.1, 2.5, 0.2)

# Prepare input
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict button
if st.button('Predict'):
    prediction = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    # Class names
    iris_classes = ['Setosa', 'Versicolor', 'Virginica']

    st.write(f"**Predicted Species:** {iris_classes[prediction]}")
    st.write("**Prediction Probabilities:**")
    for name, prob in zip(iris_classes, pred_proba):
        st.write(f"- {name}: {prob:.2f}")

# Optional: Add footer
st.write("---")
st.write("Deployed with Streamlit 🚀")
