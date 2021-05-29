from __future__ import division, print_function
from flask import Flask, request, \
        render_template, redirect, url_for,\
        session, send_file

# Flask WTF FORMS
from flask_wtf import FlaskForm,RecaptchaField
from wtforms import (StringField,SubmitField,
                     DateTimeField, RadioField,
                     SelectField,TextAreaField, DateField)

from wtforms.validators import DataRequired
import sys
import os
import glob
import re
import cv2
# import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
from PIL import Image

#keras
import tensorflow.keras as krs
# from keras import Model
# from keras.layers import Input
# from tensorflow.keras.models import load_model
# from keras.applications.resnet50 import preprocess_input, decode_predictions


#xml
from xml.etree import ElementTree


app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"

app.config["RECAPTCHA_PUBLIC_KEY"] = "6LcN1ucaAAAAACyMzmU6Tzy_DshySBfzdpxQdTHJ"
app.config["RECAPTCHA_PRIVATE_KEY"] = "6LcN1ucaAAAAAJSFWsyQjD4wq4REfhrvhftIELuw"


class Widgets(FlaskForm):
    recaptcha = RecaptchaField()

    name = StringField(label="Name", validators=[DataRequired()])

    radio = RadioField(label ="Please select Your Programming language ",
                       choices=[('Python', "Python"), ["C++","C++"]])

    submit = SubmitField(label="Submit")


@app.route("/", methods=("GET", "POST"))
def home():
    form = Widgets()
    if request.method == "POST":
        if form.validate_on_submit():
            session["name"] = form.name.data
            print("Name Entered {}".format(form.name.data))
            return redirect(url_for('result'))

    if request.method == "GET":
        return render_template("list.html", form=form)


@app.route("/result", methods=["GET", "POST"])
def result():
    return "Thanks {}".format(session["name"])
# MAIN NEIRON

SIZE = 224

train, _ = tfds.load('cats_vs_dogs', split=['train[:1%]'], with_info=True, as_supervised=True)

for img, lable in train[0].take(1):
  plt.figure()
  plt.imshow(img)
  print(lable)

def resize_image(img, lable):
  img = tf.cast(img, tf.float32)
  img = tf.image.resize(img,(SIZE, SIZE))
  img = img / 255.0
  return img, lable

train_resized = train[0].map(resize_image)
train_batches = train_resized.shuffle(1000).batch(16)

#Сверточная нейросеть
base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
base_layers.trainable = False

model = tf.keras.Sequential([
                             base_layers,
                             GlobalAveragePooling2D(),
                             Dropout(0.2),
                             Dense(1)
])
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_batches, epochs=1)

files.upload=open('C:\\screens\\1.jpg', 'r') # Загружаем свою фотку

img = load_img('golder-retriever-puppy.jpg')
img_array = img_to_array(img)
img_resized, _ = resize_image(img_array, _)
img_expended = np.expand_dims(img_resized, axis=0)
model.predict(img_expended)

files.cv2.imread('screens')

for i in range(6):
  img = load_img(f'{i+1}.jpg')
  img_array = img_to_array(img)
  img_resized, _ = resize_image(img_array, _)
  img_expended = np.expand_dims(img_resized, axis=0)
  prediction = model.predict(img_expended)[0][0]
  pred_label = 'КОТ' if prediction < 0.5 else 'СОБАКА'
  plt.figure()
  plt.imshow(img)
  plt.title(f'{pred_label} {prediction}')




# dic = {0 : 'Cat', 1 : 'Dog'}
#
# #model = cv2.imread('./static/model.h5')
# nw = 224
# nh = 224
# ncol = 3
# visible2 = krs.layers.Input(shape=(nh,nw,ncol), name = 'imginp')
# resnet = krs.applications.resnet_v2.ResNet50V2(include_top=True,
# weights='imagenet', input_tensor=visible2,
# input_shape=None, pooling=None, classes=1000)
#
#
# def predict_label(img_path):
# 	i = cv2.imread(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = resnet.predict_classes(i)
#
# 	return dic[p[0]]
#
# # routes
# @app.route("/", methods=['GET', 'POST'])
# def main():
# 	return render_template("index.html")
#
# @app.route("/about")
# def about_page():
# 	return "Please subscribe  Artificial Intelligence Hub..!!!"
#
# @app.route("/submit", methods = ['GET', 'POST'])
# def get_output():
# 	if request.method == 'POST':
# 		img = request.files['my_image']
#
# 		img_path = "static/" + img.filename
# 		img.save(img_path)
#
# 		p = predict_label(img_path)
#
# 	return render_template("index.html", prediction = p, img_path = img_path)
# file_name = 'reed_college_courses.xml'
# full_file = os.path.abspath(os.path.join('data', file_name))
#
# dom = ElementTree.parse(full_file)
#
# courses = dom.findall('course')
#
# for c in courses:
#
#     title = c.find('title').text
#     num = c.find('crse').text
#     days = c.find('days').text
#
#     print(' * {} [{}] {} *='.format(
#         num, days, title
#     ))

if __name__ == "__main__":
    app.run(debug=True)


