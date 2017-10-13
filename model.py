from __future__ import print_function
import keras
import numpy as np
import pickle 

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.utils import plot_model
from cv2 import imread

from data import getVideoData
from data import getAudioData
from data import getFusionData

def trainFinal(final):
    print('Getting data...')
    audio_video_data_tuple, label_on_correspondence= getFusionData()
    print('Data ready')

    final.fit(audio_video_data_tuple, label_on_correspondence,
            batch_size=10, epochs=20, verbose=1)

    return final

def fusionBranch(video_branch, audio_branch):

    final = Sequential()
    final.add(Merge([video_branch, audio_branch]))
    final.add(Dense(128, activation='sigmoid'))
    final.add(Flatten())
    final.add(Dense(2, activation='sigmoid'))
    final.add(Dense(2, activation='softmax'))

    final.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
            )


    return final

def getModelArchitecture(input_shape, final_pool):
    cmodel = Sequential()

    #cmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding = 'same'))
    cmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding = 'same'))
    cmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    cmodel.add(Dropout(0.25))

    cmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    cmodel.add(Dropout(0.25))

    #cmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    cmodel.add(Dropout(0.25))

    #cmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    cmodel.add(MaxPooling2D(pool_size=(final_pool), strides=None, padding='same'))
    cmodel.add(Dropout(0.25))

#    cmodel.add(Flatten())
#    cmodel.add(Dense(32, activation='sigmoid'))
#    cmodel.add(Dropout(0.25))
#    cmodel.add(Dense(2, activation='sigmoid'))

    return cmodel

def addFCLayers(cmodel):

    cmodel.add(Flatten())
    cmodel.add(Dense(32, activation='sigmoid'))
    cmodel.add(Dropout(0.25))
    cmodel.add(Dense(2, activation='sigmoid'))

    return cmodel
    

def trainModel(model, x_train, y_train, x_test, y_test):
    model.compile(loss='categorical_crossentropy', 
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

    model.fit(x_train, y_train, 
            batch_size=8,
            verbose=1,
            validation_data=(x_test, y_test))

    return model

def getTrainedModel(model, dataFactory):
    print("reading Images")
    x_train, y_train, x_test, y_test = dataFactory()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model = trainModel(model, x_train, y_train, x_test, y_test)
    score = model.evaluate(x_test, y_test, verbose=1)

    print(score)

    return model


def vidMain():
    model = getModelArchitecture((224, 224, 3), (28, 28))
    print(model.summary())
    model = getTrainedModel(model, dataFactory = getVideoData)

    return model

def audioMain():
    model = getModelArchitecture((199, 257, 3), (33, 25))
    model = addFCLayers(model)
    print(model.summary())
    model = getTrainedModel(model, dataFactory = getAudioData)

def save(model, fname):
    pickle.dump(model, open(fname, 'wb'))

def load(fname):
    return pickle.load(open(fname, 'rb'))


def fusedMain():
    vmodel = getModelArchitecture((224, 224, 3), (28, 28))
    amodel = getModelArchitecture((199, 257, 3), (25, 33))
    
    f = fusionBranch(vmodel, amodel)
    f = trainFinal(f)

    save(f, 'model.dmp')
    
    return f

if __name__ == '__main__':
    fusedMain()

