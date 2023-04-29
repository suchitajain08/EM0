from __future__ import division, print_function
#import sys
import os
import cv2
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")
    
    
@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    i=0

    GR_dict={0:(0,255,0),1:(0,0,255)}
    model = tf.keras.models.load_model('final_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output=[]
    cap = cv2.VideoCapture(0)
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while (i<=30):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        found, w = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
        
        if len(faces) == 0 and len(found) == 0:
            cv2.putText(img, "No face or object detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)

            face_img = gray[y:y+h,x:x+w]

            face_image = gray[y:y + h, x:x + w]
            face_image = cv2.resize(face_image, (48, 48))
            face_image = np.reshape(face_image, [1, 48, 48, 1]) / 255.0

        # Predict the emotion of the face using the trained model
            emotion_probabilities = model.predict(face_image)[0]
            emotion_index = np.argmax(emotion_probabilities)
            emotion_label = emotion_labels[emotion_index]
            
            
            
            
            
            cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27: 
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("buttons.html",final_output=final_output1)


@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/surprise', methods = ['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/angry', methods = ['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods = ['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods = ['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")

@app.route('/movies/happy', methods = ['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods = ['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods = ['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")

@app.route('/songs/surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songsSad.html")

@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")

@app.route('/templates/Gallary_page', methods = ['GET', 'POST'])
def Gallary():
    return render_template("Gallary_page.html")

@app.route('/templates/index', methods = ['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/templates/Support_page', methods = ['GET', 'POST'])
def Support():
    return render_template("Support_page.html")
    
if __name__ == "main":
    app.run(debug=True)