import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

st.title('Predict Skin Cancer Malignant and Benign')
# Load the pre-trained model
model = load_model('my_model.h5') 

uploaded = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

if uploaded is not None:
    # Process the uploaded image
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img = img.resize((128, 128))  # Resize the image to (128, 128)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Make a prediction
    classes = model.predict(images, batch_size=10)

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(classes)

    # Determine the class label based on the index
    if predicted_class_index == 1:
        st.write('Prediction: Malignant')
    else:
        st.write('Prediction: Benign')
