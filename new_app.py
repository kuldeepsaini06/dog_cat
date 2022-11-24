import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageOps

#@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model("my_model.hdf5")
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # DOG CAT Classification
         """
         )

file = st.file_uploader("Please upload file", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
        size = (64,64)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    #st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = predictions[0][0]
    print("This image most likely belongs to {} .".format(score))

submit= st.button("Predict")

if submit:
    
    if image is not None:
       image = Image.open(file)
       predictions = import_and_predict(image, model)
       score = predictions[0][0]
       
       
      
       if score>0.5:
           st.success("Predicted as a {}".format("DOG"))
       else:
           st.success("Predicted as a {}".format("Cat"))
    else:
        st.text("Please check the extension of image")
else:
    pass
        
