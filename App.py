import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time  # Import the time module

# Load the trained model
model = load_model('animal_classifier_model.keras')

# Define class names (same order as the model was trained)
class_names = ['🐱 Cat', '🐶 Dog', '🐍 Snake']  # Update with your actual class names

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 20px;
    }
    .header {
        font-size: 20px;
        color: #4CAF50;
        text-align: center;
    }
    .prediction {
        font-size: 24px;
        color: #0066CC;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the title of the app
st.markdown('<div class="title">🐾 Animal Image Classifier 🐾</div>', unsafe_allow_html=True)

# Allow users to upload images
st.markdown('<div class="header">Upload an image of an animal (JPG, JPEG, or PNG)</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])


def predict(image):
    # Preprocess the image
    img = image.resize((150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict the class of the image
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return predicted_class, confidence


if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True, clamp=True)

    # Progress bar for prediction
    st.write("Analyzing the image...")
    progress_bar = st.progress(0)

    for percent in range(0, 101, 10):
        progress_bar.progress(percent)
        time.sleep(0.1)  # Correct usage of time.sleep()

    # Predict and display the results
    predicted_class, confidence = predict(Image.open(uploaded_file))
    st.markdown(f'<div class="prediction">Predicted class: {predicted_class}<br>Confidence: {confidence * 100:.2f}%</div>', unsafe_allow_html=True)
