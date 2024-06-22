import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers

# Appname
st.set_page_config(page_title="Citrus Leaf Disease Classifier", layout="wide")

# Load your model and its weights
model = tf.keras.models.load_model('CNN-JerukSiamChecker.h5', compile=False)
class_names = ["Blackspot Leaf", "Canker Leaf", "Greening Leaf", "Powdery Mildew", "Citrus Leafminer", "Healthy Leaf"]

# Define the Streamlit app
def main():
    st.markdown("<h1 style='text-align: center; color: #fff;'>Citrus Leaf Disease Classifier</h1>", unsafe_allow_html=True)
    st.write("Upload an image for classification")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        image = image.resize((224, 224))
        image = np.array(image)
        image = preprocess_input(image)

        # Make predictions
        predictions = model.predict(np.expand_dims(image, axis=0))

        if np.isnan(predictions).any():
            st.write("Prediction result is NaN. Please try with another image")
        else:
            predicted_class = np.argmax(predictions)
            confidence = predictions[0][predicted_class]

            st.write(f"Predicted class: {class_names[predicted_class]}")
            st.write(f"Confidence: {confidence:.2f}")

    st.title("This model is capable of classifying:")
    for class_name in class_names:
        st.write("- " + class_name)

# Run the app
if __name__ == '__main__':
    main()
