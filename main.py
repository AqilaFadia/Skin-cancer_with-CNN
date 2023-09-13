import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

st.title('Predict Skin Cancer Malignant and Benign')
# Load the pre-trained model
model = load_model('./model/my_model.h5') 

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
# add article
st.sidebar.title('More Article')
st.sidebar.markdown(f"[Improving Employee Retention by Predicting Employee Attrition Using Machine Learning](https://medium.com/@aqilafadiamariana/improving-employee-retention-by-predicting-employee-attrition-using-machine-learning-f576bea204d8)")
st.sidebar.markdown(f"[Investigate Hotel Business using Data Visualization](https://medium.com/@aqilafadiamariana/investigate-hotel-business-using-data-visualization-cef104723962)")
st.sidebar.markdown(f"[Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning](https://medium.com/@aqilafadiamariana/predict-customer-personality-to-boost-marketing-campaign-by-using-machine-learning-79368c2dc87d)")
st.sidebar.markdown(f"[Classification of Skin Cancer model using Tensorflow](https://medium.com/@aqilafadiamariana/classification-of-skin-cancer-model-using-tensorflow-19ff2c000087)")
st.sidebar.markdown(f"[Predict Credit Scores by Enhancing Precision Function](https://medium.com/@aqilafadiamariana/predict-credit-scores-by-enhancing-precision-function-a0bb104528d)")
