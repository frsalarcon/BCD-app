import numpy as np
import streamlit as st
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from keras.preprocessing import image
import pandas as pd
from tensorflow.keras import layers
from skimage import*
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from glob import glob
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from skimage import io
import matplotlib.image as mpimg
from PIL import Image
from io import BytesIO
# ---------------------------------- modelo ---------------------------------- #

modelo='model.h5'
pesos_modelo='pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)

# -------------------------------- predicción -------------------------------- #

prediction=[]

#predict('benigno_t.png')
import streamlit as st
import numpy as np
from PIL import Image
import datetime

def main():

    bcd = Image.open('BCD.png')
    st.image(bcd)
    st.text('Ingresar datos del usuario:')
    @st.cache(allow_output_mutation=True)
    def get_data():
        return []

    @st.cache(allow_output_mutation=True)
    def get_data():
        return []

    name = st.text_input("Nombre del usuario")
    appe = st.text_input("Apellido del usuario")
    # st.title("BCD - Breast Cancer Detection.")
    # nombre= st.text_input("Nombres del paciente:")
    # apellido=st.text_input("Apellidos del paciente:")
    # contacto=st.text_input("Contacto:")
    # comentario=st.text_input("Comentario:")
    # date = st.date_input("Ingresar fecha de análisis.")
    # frame=(nombre,apellido,contacto,comentario,date)
    # if st.button("Generar datos:"):
    #     frame.append({"Nombres": nombre, "Apellidos": apellido, "Fecha de análisis":date ,"Contacto":contacto , "Comentario":comentario})
    #     st.write(pd.DataFrame(frame()))

    img_file_buffer = st.file_uploader("Carga una imagen de muestra histologica.", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            st.image(image, caption="Muestra lista para el análisis.", use_column_width=False)


    if st.button("Ejecutar predicción."):
        im2arr = np.array(image)
        im2arr.resize((100,100,3), refcheck=True)
        x = np.expand_dims(im2arr, axis=0)
        array = model.predict(x)
        result = array[0]
        answer = np.argmax(result)
        if answer == 0:
            answer="Predicción de la muesta: Benigna."
        elif answer == 1:
            answer="Predicción de la muesta: Maligna."
        st.success(answer)
if __name__ == '__main__':
    main()