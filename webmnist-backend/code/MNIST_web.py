# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:30:15 2018

@author: Pankaj Giri
"""

import os
from flask import Flask
from keras.models import load_model
from flask import request, flash, redirect, url_for
import base64
from flask_cors import CORS
import io
from PIL import Image
from keras import backend as K
import numpy as np
import cv2
import imutils

app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default

PATH_TO_MODEL = '../models/model_mnist_keras1.ckpt'
UPLOAD_FOLDER = '../upload/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_model_from_path(PATH):
    print('Loading model from path', PATH)
    model = load_model(PATH)
    return model


def crop_image(image):
    # Mask of non-black pixels (assuming image has a single channel).
    mask = image > 0
    tol = 15

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    print(mask.shape)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box.
    return image[x0 - tol:x1 + tol, y0 - tol:y1 + tol]


def convert_image(input_):
    input_ = input_.convert('1')
    input_ = np.array(input_.getdata()).reshape(input_.size[1], input_.size[0])

    print("Shape of the input array :", input_.shape)
    print("Unique :", np.unique(input_))
    X = input_.copy()
    X = 255 - X

    image = crop_image(X)
    tmp_image = Image.fromarray(np.uint8(image))

    tmp_image = tmp_image.resize((28, 28), Image.ANTIALIAS)

    print("After resizing :", tmp_image.size[1], tmp_image.size[0])
    image = np.array(tmp_image.getdata()).reshape(tmp_image.size[1], tmp_image.size[0])

    #Image.fromarray(np.uint8(image)).show()
    return image.reshape(1, 28, 28, 1)


class WebMNIST:
    def __init__(self, PATH):
        #        Otherwise throws a "Cannot interpret feed_dictk key as Tensor" error
        K.clear_session()
        self.model = load_model_from_path(PATH)

    def predict(self, X):
        if self.model != None:
            processed_image = convert_image(X)
            prediction = self.model.predict(processed_image)
            number = np.argmax(prediction)

        #    The return type needs to be a string
        return str(number)

def segment_predict_digits(image):
    mnist = WebMNIST(PATH_TO_MODEL)
    predictions = list()
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_scale_image = cv2.GaussianBlur(gray_scale_image, (5, 5), 0)

    # threshold the image
    _, threshold_image = cv2.threshold(gray_scale_image, 127, 255, cv2.THRESH_BINARY_INV)

    # dilate the white portions
    dilated_image = cv2.dilate(threshold_image, None, iterations=2)

    # find contours in the image
    contours = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    i = 0

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        roi = image[y:y + h, x:x + w]

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Made Border-Size since model had been trained with letters present at center of image
        border_size = 40
        roi = cv2.copyMakeBorder(roi,top=border_size,bottom=border_size,left=border_size,right=border_size,borderType=cv2.BORDER_CONSTANT,value=[255,255,255])

        # Convert image to PIL to pass to model
        roi  = Image.fromarray(roi)
        predictions.append(mnist.predict(roi))

        i = i + 1
    return predictions

@app.route('/predict_input', methods=['GET', 'POST'])
def predict_input():
    print('In predict input method')

    if request.method == 'POST':
        image_bytes = io.BytesIO(base64.b64decode(request.form['file']))
        image = Image.open(image_bytes)
        # convert to opencv-python
        opencv_image = np.array(image)[:, :, ::-1].copy()
        predictions = segment_predict_digits(opencv_image)
        print("Predictions:",predictions)
        result = " , ".join(str(prediction) for prediction in predictions)
        return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
