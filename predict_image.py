import numpy as np
import cv2 as cv
from keras.models import load_model
from PIL import Image

model = load_model('twof.h5')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
colors = {'neutral':(255, 255, 255), 'angry':(0, 0, 255), 'fear':(0, 0, 0), 'happy':(0, 255, 255), 'sad':(255, 0, 0), 'surprised':(255, 245, 0)}
imotions = {0:'angry', 2: 'happy', 3:'sad', 4:'surprised', 5:'neutral'}

def convert_dtype(x):
    x_float = x.astype('float32')
    return x_float

def normalize(x):
    x_n = (x - 0) / (255)
    return x_n

def reshape(x):
    x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    return x_r

def get_emotion(file):
    fr = Image.open(file).convert("RGB")
    fr = np.array(fr)
    gray = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        print("HIII=> in get_emotion => forLoop")
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = fr[y:y + h, x:x + w]
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation = cv.INTER_AREA)
        roi_gray = convert_dtype(np.array([roi_gray]))
        roi_gray = normalize(roi_gray)
        roi_gray = reshape(roi_gray)
        pr = model.predict(roi_gray)[0]
        # cv.rectangle(fr, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        maxindex = int(np.argmax(pr))
        print("IMOTIONS", imotions[maxindex])
        return str(imotions[maxindex])
        # cv.putText(fr, imotions[maxindex], (x + 20, y - 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    print("HIII=> in get_emotion => outside forLoop")
    return str(imotions[5])
    # cv.imshow('img', fr)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    