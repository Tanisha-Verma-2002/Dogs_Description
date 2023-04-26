import streamlit as st
import sklearn
import pandas as pd
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.activations import softmax
from keras import preprocessing
from keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from keras.models import Sequential
from keras.layers import Dropout, Dense,BatchNormalization, Flatten, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.layers import Conv2D, Reshape
import h5py

st.header('Image Class Predictor')

def main():
    file_uploaded = st.file_uploader('Choose the File' , type = ['jpg','png','jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    classifier_model = tf.keras.models.load_model('model.hdf5')
    shape = ((384, 384, 3))
    shape_im= ((224,224,3))
    tf.keras.Sequential([hub.KerasLayer(classifier_model,input_shape = shape_im)])
    test_image = image.resize((224,224))
    test_image = np.array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image,axis=0)
    class_names = ['angry','happy','relaxed','sad']
    predictions = classifier_model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The Predicted Image Shows the Emotion: {}".format(image_class)
    return result

if __name__ == '__main__':
    main()

