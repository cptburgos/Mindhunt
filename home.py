import streamlit as st
import os
from PIL import Image
from numpy import asarray

from tensorflow import keras


BASEDIR = os.path.abspath(os.path.dirname(__file__))
MODELDIR = os.path.join(BASEDIR, 'models')

# Code for markdowns
BOOTSTRAP = (
    '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/'
    'bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/'
    'azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" '
    'crossorigin="anonymous">'
)

BANNER = """
<div class="text-center" style = "background-color : #FF4040">
  <h1>Applaudo Studios</h1>
  <h4>Image classifier</h4class=>
</div>
"""

FOOTER = """
<footer class="text-center text-lg-start" style="background-color :#FF4040">
  <!-- Copyright -->
  <div class="text-center p-3" style="background-color: rgba(0, 0, 0, 0.2);">
    This MVP has been developed for this Mindhunt by
    <a class="text-dark" href="https://applaudo.com/">
      Applaudo Studios
    </a>
    © 2023 Copyright
  </div>
  <!-- Copyright -->
</footer>"""

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