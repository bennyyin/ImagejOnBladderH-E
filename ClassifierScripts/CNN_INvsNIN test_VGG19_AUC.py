# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:47:20 2019

@author: Benny  Yin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:44:31 2018

@author: Benny  Yin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:07:18 2018

@author: Benny  Yin
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import backend as K
from keras import applications



# dimensions of our images.
img_width, img_height = 175, 175
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'C:/Life/Thesis/data/train/';validation_data_dir = 'C:/Life/Thesis/data/validation/'
#train_data_dir = 'C:/Life/Thesis/data/train_process/';validation_data_dir = 'C:/Life/Thesis/data/validation_process/'
#train_data_dir = 'C:/Life/Thesis/data/train_redCNT/';validation_data_dir = 'C:/Life/Thesis/data/validation_redCNT/'

train_IN_num, train_NIN_num = 399, 384
validation_IN_num, validation_NIN_num = 104, 76

nb_train_samples = train_IN_num + train_NIN_num
nb_validation_samples = validation_IN_num + validation_NIN_num
epochs = 30
batch_size = 9

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    #datagen = ImageDataGenerator()
    #datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,vertical_flip=True)
    # build the VGG16 network
    model = applications.VGG19(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve

def train_top_model():
    X_train = np.load(open('bottleneck_features_train.npy','rb'))
    #train_labels = np.array([0] * int((nb_train_samples / 2)) + [1] * int(nb_train_samples / 2))
    Y_train = np.array([0] * train_IN_num + [1] * train_NIN_num)

    X_test = np.load(open('bottleneck_features_validation.npy','rb'))
    #validation_labels = np.array([0] * int((nb_validation_samples / 2)) + [1] * int((nb_validation_samples / 2)))
    Y_test = np.array([0] * validation_IN_num + [1] * validation_NIN_num)
    
    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size, verbose=1)
    model.save_weights(top_model_weights_path)
    
    y_pred_keras = model.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
          #validation_data=(X_test, Y_test))   
    return fpr_keras, tpr_keras, thresholds_keras

save_bottlebeck_features()
fpr_keras, tpr_keras, thresholds_keras = train_top_model()

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--',)
plt.plot(fpr_keras, tpr_keras, color='g',label='CNN_VGG19 (area = {:.3f})'.format(auc_keras))     

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
