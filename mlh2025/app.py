import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load trained model
model = load_model("iris_model")

# Streamlit UI
st.title("Iris Species Prediction App 🌸")
st.write("Enter the values and get predictions!")

# User input fields
sepal_length = st.number_input("Sepal Length", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.1, max_value=3.0, step=0.1)

# Make prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    prediction = predict_model(model, data=input_data)
    species = prediction["prediction_label"].values[0]
    st.success(f"Predicted Species: **{species}**")
