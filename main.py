import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the model
model_path = os.path.join('models', 'imageclassifier.h5')
model = load_model(model_path)

def predict(image):
    # Preprocess the image
    resize = cv2.resize(image, (256, 256))
    input_image = np.expand_dims(resize / 255.0, 0)

    # Get the prediction
    prediction = model.predict(input_image)[0][0]

    # Define the classes
    classes = ['Happy', 'Sad']

    # Determine the class based on the prediction
    predicted_class = classes[int(np.round(prediction))]
    confidence = float(prediction)

    return predicted_class, confidence

def main():
    st.set_page_config(layout="wide")  # Set wide layout to fit content in one page
    st.title("Image Classifier")
    st.write("A Deep CNN Image Classifier is a powerful computer program that can look at pictures and identify what's in them. It uses a specialized network of interconnected layers to learn and recognize different features in images, like shapes and patterns. This technology has been really successful in tasks like telling apart different animals or objects in photos.")

    # Create a layout with two columns
    col1, col2 = st.columns([1, 3])  # Reverse the order of columns

    # Column 1 for emoji and confidence
    with col1:
        uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

            predicted_class, confidence = predict(image)

            emoji_size = 80  # Adjust the emoji size
            if predicted_class == 'Happy':
                st.markdown(f'<p style="font-size: {emoji_size}px;">Emoji: 😀</p>', unsafe_allow_html=True)  # Happy emoji
            else:
                st.markdown(f'<p style="font-size: {emoji_size}px;">Emoji: 😢</p>', unsafe_allow_html=True)  # Sad emoji

            st.write(f"Confidence: {confidence:.2f}")

    # Column 2 for uploaded image and output
    with col2:
        if uploaded_image is not None:
            st.image(image, channels="BGR", caption="Uploaded Image", width=300)  # Adjust the width here

            st.write(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
