from cv2 import imread
import numpy as np
from keras import utils
import cv2

def data_generator(batch_size):
    yield 0

def readImages(dir, y, limit, data, labels, ext):
    for _ in list(range(1, limit + 1)):
        src = dir + 'image (' + str(_) + ')' + ext

        if 'audioMMMM' in dir:
            data.append(imread(src, cv2.IMREAD_GRAYSCALE))
        else:
            data.append(imread(src))

        labels.append(y)

    return data, labels


def getFusionData():

    images, _, _, _ = getVideoData()

    all_images = np.append(images, images)
    all_images = all_images.reshape((1600, 224, 224, 3))
    
    
    pair_labels = []
    for _ in range(800):
        pair_labels.append(0)

    print(all_images.shape)

    data = labels = []

    audio1, _, _, _ = getAudioData()
    audio2, _, _, _ = getAudioData('.\\data\\audio\\motorbikes\\', '.\\data\\audio\\airplanes\\' )

    all_audio = np.append(audio1, audio2)

    all_audio = all_audio.reshape((1600, 199, 257, 3))

    for _ in range(800):
        pair_labels.append(1)


    pairs = [all_images, all_audio]

    pair_labels = np.array(pair_labels)
    #pair_labels = utils.to_categorical(pair_labels, 2)

    print(pairs[0].shape, pair_labels.shape)
    return pairs, pair_labels


def getAudioData(dir1 = '.\\data\\audio\\airplanes\\' , dir2 = '.\\data\\audio\\motorbikes\\'):
    data = []
    labels = []

    data, labels = readImages(dir1, 0, 400, data, labels, '.jpg')
    data, labels = readImages(dir2, 1, 400, data, labels, '.jpg')

    train_data = data[:800]
    train_label = labels[:800]

    test_data = data[100:110]
    test_label = labels[100:110]

    test_label = utils.to_categorical(test_label, 2)
    train_label = utils.to_categorical(train_label, 2)
    
    train_data = np.array(train_data)
    #train_label = np.array(train_label)

    test_data = np.array(test_data)
    #test_label = np.array(test_label)
    
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    train_data /= 255
    test_data /= 255

    return train_data, train_label, test_data, test_label

def getVideoData(dir1 = '.\\data\\video\\airplanes\\', dir2 = '.\\data\\video\\motorbikes\\'):
    data = []
    labels = []

    data, labels = readImages(dir1, 0, 400, data, labels, '.jpg')
    data, labels = readImages(dir2, 1, 400, data, labels, '.jpg')

    train_data = data[:800]
    train_label = labels[:800]

    test_data = data[100:110]
    test_label = labels[100:110]

    test_label = utils.to_categorical(test_label, 2)
    train_label = utils.to_categorical(train_label, 2)
    
    train_data = np.array(train_data)
    #train_label = np.array(train_label)

    test_data = np.array(test_data)
    #test_label = np.array(test_label)
    
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    train_data /= 255
    test_data /= 255

    return train_data, train_label, test_data, test_label
