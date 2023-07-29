import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
import time

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'imageclassifier.h5')
model = tf.keras.models.load_model(model_path)

# Define the classes
classes = {0: 'Happy', 1: 'Sad'}

# Emojis
emoji_happy = "😊"
emoji_sad = "😢"

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, (256, 256))
    image = np.expand_dims(image / 255.0, axis=0)
    return image

def main():
    st.title('Image Classifier - Happy or Sad')

    # Create a layout with two columns
    col1, col2 = st.columns([2, 1])  # Ratio: 2/3 for input column, 1/3 for output column

    # Upload image in the left column
    with col1:
        uploaded_file = st.file_uploader("Upload an image, and the model will predict whether it's happy or sad!",
                                         type=["jpg", "jpeg", "png"], key="fileUploader")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

    if uploaded_file is not None:
        # Preprocess the uploaded image
        image_np = np.array(image)
        resized_image = preprocess_image(image_np)

        # Display clean black background with output text and emojis in the right column
        with col2:
            # Set CSS styling to position the processing text at the top right corner
            st.markdown("<style>div.stButton > button:first-child {float: right;}</style>", unsafe_allow_html=True)
            st.info("Processing...")

        # Introduce a delay to create suspense (you can adjust the delay time as per your preference)
        time.sleep(4)

        # Make prediction
        prediction = model.predict(resized_image)

        # Get the predicted class
        predicted_class = classes[int(np.round(prediction[0][0]))]

        # Display clean black background with output text and emojis in the right column
        with col2:
            st.markdown("<div style='background-color: black; padding: 20px; text-align: center;'>"
                        f"<h1 style='color: white; font-size: 40px;'>Prediction: {predicted_class}</h1>"
                        "<br>"
                        f"<p style='font-size: 80px;'>{emoji_happy if predicted_class == 'Happy' else emoji_sad}</p>"
                        "</div>", unsafe_allow_html=True)

            # Small text at bottom right corner
            st.markdown("<div style='position: fixed; bottom: 10px; right: 10px; color: gray;'>"
                        "by JALLA LABS PRIVATE LIMITED</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
