import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2
import streamlit as st

# Load the trained model
MODEL_PATH = "mobilenet_flower_model.h5"  # Change this to your actual model path
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

# Define class names (ensure this matches your dataset labels)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Image preprocessing function
def preprocess_image(image):
    IMAGE_SIZE = 224
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Streamlit app
st.title("Flower Classification App 🌸")
st.write("Upload an image of a flower, and the AI will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    st.write(f"### Prediction: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}")