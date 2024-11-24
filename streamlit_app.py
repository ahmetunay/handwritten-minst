# libraries
import os
import cv2 as cv
import numpy as np
import tensorflow as tf

import streamlit as st
from PIL import Image

import tempfile
from tensorflow.keras.models import load_model

# function 1 : to classify the image
def classify_digit(model, image):
    img = cv.imread(image)[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    return prediction

# function 2 : to resize the image
def resize_image(image, target_size):
    img = Image.open(image)
    resized_image = img.resize(target_size)
    return resized_image

# page name
st.set_page_config(page_title='Digit Recognition', page_icon='🔢')

# example of the title, markdown, etc
st.title('Handwritten Digit Recognition 🔢')

st.markdown(r'''This simple application is designed to recognize a number from 0-9 from a PNG file with a resolution of 28x28 pixels. 
            While it may not achieve 100% accuracy, its performance is consistently high.''')
st.subheader('Have fun giving it a try!!! 😊')  

uploaded_image = st.file_uploader('Insert a picture of a number from 0-9', type='png')

if uploaded_image is not None:
    # Convert uploaded image to numpy array
    image_np = np.array(Image.open(uploaded_image))

    # Save the temporary image
    temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_image.png')
    cv.imwrite(temp_image_path, image_np)

    # Resize the image for display
    resized_image = resize_image(uploaded_image, (300, 300))

    # Display the image in the center column
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(resized_image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    submit = st.button('Predict')

    if submit:
        # Load the model
        model = load_model('BenimModelim.h5')

        # Use the model to predict the digit
        prediction = classify_digit(model, temp_image_path)
        st.subheader('Prediction Result')
        st.success(f'The digit is probably a {np.argmax(prediction)}')

    # Remove the temporary image
    os.remove(temp_image_path)
