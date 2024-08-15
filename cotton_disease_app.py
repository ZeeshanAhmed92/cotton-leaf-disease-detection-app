import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import random
from PIL import Image, UnidentifiedImageError
import numpy as np

# Custom CSS to adjust the margins
custom_css = """
    <style>
        .main .block-container {
            padding-left: -1rem;
            padding-right: -1rem;
        }
    </style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Define the header text with custom color using HTML
header_html2 = """
    <h1 style="color: lightgreen;">WELCOME TO COTTON LEAF CLASSIFICATION APP</h1>
"""

# Render the HTML in Streamlit
st.markdown(header_html2, unsafe_allow_html=True)

header_html = """
    <h2 style="color: lightgreen;">IMAGE CLASSIFICATION WITH CNN</h2>
"""

# Render the HTML in Streamlit
st.markdown(header_html, unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)

# Loading model
model = load_model('cotton_disease_detection.h5')

# Function for fetching sample images from each class
def get_sample_images(main_dir, num_samples=3):
    subdirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    sample_images = {}
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        images = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        sampled_images = random.sample(images, num_samples)
        sample_images[subdir] = [os.path.join(subdir_path, img) for img in sampled_images]
    return sample_images

# Function for displaying sample images from each class
def main():
    
    main_directory = "Cotton Disease/train"  # Replace with the path to your main directory
    sample_images = get_sample_images(main_directory)

    for folder, images in sample_images.items():
        header_html1 = f"""
        <h3 style="color: lightgreen;">{folder} Samples</h3>
                                                        """
        # Render the HTML in Streamlit
        st.markdown(header_html1, unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, img_path in enumerate(images):
            image = Image.open(img_path)
            cols[idx].image(image, use_column_width=True)

if __name__ == "__main__":
    main()
      
# Function for preprocessing uploaded image           
def image_preprocessing(image):
    # image = Image.open(image)
    image = image.resize((256,256),Image.Resampling.NEAREST)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array/255
    return image_array

# uploaded file    
uploaded_file = st.file_uploader('Upload an Image for Classification', type=['jpg','jpeg','png'])

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, use_column_width=True, caption='Uploaded Image')
        processed_image = image_preprocessing(uploaded_image)
        
        if st.button('Predict'):
            prediction = model.predict(processed_image)
            class_names = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
            string = 'This Image most likely is: ' + class_names[np.argmax(prediction, axis=1)[0]]
            st.success(string)
    
            probability = np.max(prediction)  # Get the highest probability value
            # Display the result
            st.success(f"Predicted Class: {class_names[np.argmax(prediction, axis=1)[0]]} with Probability: {probability:.2f}")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a file in JPG, JPEG, or PNG format.")
else:
    st.write("Please upload an image file.")
    