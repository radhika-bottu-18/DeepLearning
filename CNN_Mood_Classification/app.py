import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io

model = load_model('mood_model.h5')

st.title(':smile: Mood Classification Ap(Happy or Sad)')
st.markdown('Upload an image, and the model will classify the mood.')

uploaded_file = st.file_uploader('Choose an image...',type=['jpg','jpeg','png'])

if uploaded_file is not None:
    #Display the image
    image = Image.open(uploaded_file)
    st.image(Image,caption="Uploaded Image",use_column_width=True)
    #Preprocessing the image
    img = image.resize((64,64))
    img_array= img_to_array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)
   
   #predict  
    prediction = model.predict(image_array)[0][0]
    mood = "Happy :smile: " if prediction >0.5 else "Sad :)"

    st.markdown(f'### Predicted Mood: {mood}')