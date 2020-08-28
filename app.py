from flask import Flask, render_template, request, send_from_directory
import numpy as np
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras
from keras.models import load_model

model = load_model('./models/model_weights.h5')
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')

COUNT = 0
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save(f'static/{COUNT}.jpg')
    img_arr = cv2.imread(f'static/{COUNT}.jpg')
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_arr)     #gives bounding box coordinates
    crop_img = img_arr[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]

    img_arr = cv2.resize(crop_img, (48,48))
    img_arr = img_arr.reshape(1, 48, 48, 1)
    preds = model.predict(img_arr)[0]
    preds = [round(x*100,2) for x in preds]

    COUNT += 1
    return render_template('prediction.html', data=preds)

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))

if __name__ == '__main__':
    app.run(debug=True)
