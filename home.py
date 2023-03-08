import streamlit as st
import os
from PIL import Image
from numpy import asarray
from constants import (BOOTSTRAP, BANNER, FOOTER)
from tensorflow import keras


BASEDIR = os.path.abspath(os.path.dirname(__file__))
MODELDIR = os.path.join(BASEDIR, 'models')



st.markdown(BOOTSTRAP, unsafe_allow_html=True)
st.markdown(BANNER, unsafe_allow_html=True)

# Loading the AI model
model = keras.models.load_model(os.path.join(MODELDIR, 'cifar10_02.h5'))

st.write("Hello, welcome to the image classifier, sponsored by Applaudo Studios")

img_uploader = st.file_uploader("Choose the image to classify")

if img_uploader != None:

  image_1 = Image.open(img_uploader)
  
  image_small = image_1.resize((32,32), resample = Image.BICUBIC).convert('RGB')
  
  st.write("Image with original size:")
  st.image(image_1)
  
  st.write("Image resized for the model input:")
  st.image(image_small)
  
  class_names =['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
  
  model_input = asarray(image_small)
  
  model_input = model_input.reshape(1,32,32,3)
    
  results = model.predict(model_input)
  
  st.write(f"The image you uploaded has been classified as: {class_names[results.argmax()]}")
else:
  st.write("Here you will see the image")

st.markdown(FOOTER, unsafe_allow_html=True)