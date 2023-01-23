from tensorflow import keras
from keras.preprocessing import image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np


#keras.applications.resnet50.ResNet50

#VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

#st.text("Hello Welcome")
detector = MTCNN()
model = keras.applications.resnet50.ResNet50(
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg',
    weights='imagenet'
)
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

filenames = [sub.replace('/kaggle/input/bollywood-celeb-localized-face-dataset/', 'https://technirmitisoftwares.com/img_data/data/') for sub in filenames]

def save_uploaded_image(uploaded_image):
    try:
        with open(uploaded_image.name, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False


def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = keras.applications.resnet50.preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

    

st.title('Which bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        st.header("Image Uploded!, Processing...")
        
        # extract the features
        features = extract_features(uploaded_image.name, model, detector)
        
        # recommend
        index_pos = recommend(feature_list,features)
        predicted_actor = filenames[index_pos]
        
        # display
        display_image_resized = display_image.resize((150,150))
        col1,col2 = st.columns(2)

        with col1:
            st.subheader('Your uploaded image')
            st.image(display_image_resized,width=150, caption='Uploaded Image')
        with col2:
            st.subheader("Look Like: " + predicted_actor.split("/")[7])
            st.image(filenames[index_pos],width=150, caption='Predicted Look Alike Image')
