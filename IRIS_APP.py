import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Iris Flower Prediction", page_icon="🌸", layout="wide")

# Load the model
model = pickle.load(open("iris_model.pkl", "rb"))

# Background image
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("setosa.jpg");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🌺 Iris Flower Species Prediction 🌺")
st.markdown("Welcome to the Iris Flower Prediction App! 🌼")

# Input features
st.header("Input Features")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width]
})

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"The predicted species is: **{species[prediction[0]]}** 🌷")

# Add some visuals
st.header("Flower Characteristics Visualization")
# Load and display an example image of the Iris species
image_path = "iris_setosa.jpg"  # Replace with your image path
st.image(image_path, caption="Example of Iris Flower", use_column_width=True)

# Conclusion
st.markdown("Thank you for using the Iris Flower Prediction App! 🌼🌺🌸")




