
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import pandas as pd 
import webbrowser
from PIL import Image, ImageOps
import random

def cal_confi(State):
  global event_sum
  data = pd.read_csv("weights_data/Data.csv")                           
  df = pd.DataFrame(data)
  loc_state = df.loc[df['State'] == State] 
  event = np.array(loc_state)
  event_sum = (event[0][1]+event[0][2])

def calc_result(score):
   global total_score
   if result == "Crack":
       crack_score = random.randint(10, 25)
       total_score = crack_score + score
       if st.button('View Results'):
            if result == "Crack":
                st.error("Crack Detected")
                st.error(f"The life expectancy of this structure in {State} is {total_score}%")
                st.error("The Crack present can be harmful, and needs to be corrected as soon as possible.")

   elif result == "Not-Crack":
       crack_score = random.randint(25, 40)
       total_score = crack_score + score
       if st.button('View Results'):
            st.success("No Crack Detected")
            st.success(f"The life expectancy of this structure in {State} is {total_score}%")
            st.success("There is no such harmful crack present, and the structure is strong enough physically.")


st.markdown("""
<style>
.medium-button {
   color:white
}
body {
  color: black;
 
}
</style>
    """, unsafe_allow_html=True)
page_bg_img = '''
<style>
body {
background-image: url("https://images.pexels.com/photos/2098428/pexels-photo-2098428.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260");
background-size: cover;
}
.markdown-text-container{
  color:white !important;
}
.stMarkdown{
  color:white;
}
.stFileUploader label{
 color:white;
}
.stSelectbox label,.uploadedFileName{
 color:white; 
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Welcome to the Structural Defect Detection Program!")
st.header("Please fill the following details properly:")
State = st.selectbox(
    'Select State Name',
    ('Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'New Delhi', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Andaman and Nicobar', 'Chandigarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Jammu and Kashmir', 'Lakshadweep', 'Puducherry', 'Ladakh')
)

file = st.file_uploader("Please upload your image here:")


 
def upload_and_predict(image_data,weight):
   global prediction
   size = (224,224)
   image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
   img  = np.array(image)
   img_reshape = img[np.newaxis,...]
   prediction = weight.predict(img_reshape)
   return prediction

if file is  None:
   st.error("Please upload an image")
else:   
   try:
     image = Image.open(file)
     rebuild_model = load_model("weights_data/my_model.h5")
     rebuild_model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
     upload_and_predict(image,rebuild_model)
     class_name = ['Crack','Not-Crack']
     result = class_name[np.argmax(prediction)]
     cal_confi(State)
     calc_result(event_sum)
   except ValueError:
      st.warning("Please upload a valid image")

