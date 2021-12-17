import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as  pd

from google.colab import drive
drive.mount('/content/gdrive')

csv_file = pd.read_csv("/content/gdrive/My Drive/Emotion Detection/fer2013.csv")

len(csv_file)

len(csv_file.loc[csv_file['pixels'] != 1])

csv_file = csv_file.loc[csv_file['emotion'] != 1]
faces = csv_file['pixels'].tolist()
emotions = csv_file['emotion'].tolist()

print(emotions)

for i in range(len(emotions)):
  if emotions[i] != 0:
    emotions[i] -= 1

for i in range(len(faces)):
  faces[i] = np.array(faces[i].split()).reshape(48, 48).astype('float32')

X = np.array(faces)
y = np.array(emotions)

print(type(X))

X = faces
y = emotions

len(emotions)

#imotions = {'AN':0, 'FE':1, 'HA':2, 'SA':3, 'SU':4, 'NE':5, 'DI':6}
imotions = {'AN':0, 'HA':3, 'SA':4, 'SU':5,'NE':6}

len(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10)
X_train1, X_test1, y_train1, y_test1 = X_train, X_test, y_train, y_test

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

def convert_dtype(x):
    x_float = x.astype('float32')
    return x_float

X_train = convert_dtype(X_train)
X_test = convert_dtype(X_test)

from google.colab.patches import cv2_imshow
cv2_imshow(X_test1[5])
print(y_test1[5])

def normalize(x):
    x_n = (x - 0)/(255)
    return x_n
    
X_train = normalize(X_train)
X_test = normalize(X_test)

def reshape(x):
    x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    return x_r

X_train = reshape(X_train)
X_test = reshape(X_test)

def oneHot(y, Ny):
    y_oh = np.zeros((len(y), Ny))
    for i in range(len(y)):
      y_oh[i][y[i]] = 1
    return y_oh

y_train = oneHot(y_train, 7)
y_test = oneHot(y_test, 7)

from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
import tensorflow as tf
def create_model():
    def big_XCEPTION(input_shape, num_classes):
      img_input = Input(input_shape)
      x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
      x = BatchNormalization(name='block1_conv1_bn')(x)
      x = Activation('relu', name='block1_conv1_act')(x)
      x = Conv2D(64, (3, 3), use_bias=False)(x)
      x = BatchNormalization(name='block1_conv2_bn')(x)
      x = Activation('relu', name='block1_conv2_act')(x)

      residual = Conv2D(128, (1, 1), strides=(2, 2),
                        padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
      x = BatchNormalization(name='block2_sepconv1_bn')(x)
      x = Activation('relu', name='block2_sepconv2_act')(x)
      x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
      x = BatchNormalization(name='block2_sepconv2_bn')(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      residual = Conv2D(256, (1, 1), strides=(2, 2),
                        padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = Activation('relu', name='block3_sepconv1_act')(x)
      x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
      x = BatchNormalization(name='block3_sepconv1_bn')(x)
      x = Activation('relu', name='block3_sepconv2_act')(x)
      x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
      x = BatchNormalization(name='block3_sepconv2_bn')(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])
      x = Conv2D(num_classes, (3, 3),
                # kernel_regularizer=regularization,
                padding='same')(x)
      x = GlobalAveragePooling2D()(x)
      output = Activation('softmax', name='predictions')(x)

      model = Model(img_input, output)
      return model
    model = big_XCEPTION((48, 48, 1), 7)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False, name='Adam')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model
model = create_model()

model.summary()

len(model.layers)

history = model.fit(X_train, y_train, validation_split = 0.1, epochs=50, batch_size=32)

def predict(x):
    y = model.predict(x)
    y1 = np.zeros(y.shape)
    for i in range(len(y)):
      y1[i][np.argmax(y[i])] = 1
    y = y1
    return y

def oneHot_tolabel(y):
    y_b = []
    for i in range(len(y)):
      y_b.append(np.argmax(y[i]))
      
    y_b = np.array(y_b)
    return y_b

def create_confusion_matrix(true_labels, predicted_labels):
    cm = np.zeros((7, 7))
    for i in range(len(true_labels)):
      if true_labels[i] == predicted_labels[i]:
        cm[true_labels[i]][true_labels[i]] = cm[true_labels[i]][true_labels[i]] + 1
      elif true_labels[i] != predicted_labels[i]:
        cm[true_labels[i]][predicted_labels[i]] += 1
    return cm

predicted_labels_train = predict(X_train)

oneHot_tolabel(predicted_labels_train)

len(predicted_labels_train)

cm = create_confusion_matrix(oneHot_tolabel(y_train), oneHot_tolabel(predict(X_train))).astype(int)

print(cm)

import seaborn as sns
sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')

print(cm)

history.history.keys()
import matplotlib.pyplot as plt
plt.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'])
plt.show()

def accuracy(x_test, y_test, model):
    acc = model.evaluate(x_test, y_test)[1]
    return acc

acc = accuracy(X_test, y_test, model)
print('Test accuracy is, ', acc*100, '%')

from sklearn import metrics
print(metrics.confusion_matrix(oneHot_tolabel(y_train), oneHot_tolabel(predict(X_train)), labels=[0, 2, 3, 4, 5]))
print(metrics.classification_report(oneHot_tolabel(y_train), oneHot_tolabel(predict(X_train)), labels=[0, 2, 3, 4, 5]))

model.save('/content/gdrive/MyDrive/Emotion Detection/Untitled folder/trained_big.h5')

i = 7
from google.colab.patches import cv2_imshow
cv2_imshow(X_test1[i])



pr = predict(np.array(X_test))

oneHot_tolabel(pr) == oneHot_tolabel(y_test)

oneHot_tolabel(y_test)

#%cd '/content/gdrive/My Drive/Emotion Detection/Untitled folder/assets'

'''%cd '/content/gdrive/My Drive/Emotion Detection/Untitled folder/'
from tensorflow.keras.models import load_model
new_model = load_model('trained')
print(new_model)'''

#!python -c "import keras; print(keras.__version__)"

#!python -c "import tensorflow as tf; print(tf.__version__)"