from cv2 import imread
import numpy as np
from keras import utils
import cv2

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

    images, _, v_images, _ = getVideoData()

    all_images = np.append(images, images)
    all_v_images = np.append(v_images, v_images)

    all_images = all_images.reshape((1600, 224, 224, 3))
    all_v_images = all_v_images.reshape((200, 224, 224, 3))
    
    
    pair_labels = []
    for _ in range(800):
        pair_labels.append(0)


    print(all_images.shape)

    audio1, _, vaudio1, _ = getAudioData()
    audio2, _, vaudio2, _ = getAudioData('.\\data\\audio\\motorbikes\\', '.\\data\\audio\\airplanes\\' )

    all_audio = np.append(audio1, audio2)
    all_vaudio = np.append(vaudio1, vaudio2)

    all_audio = all_audio.reshape((1600, 199, 257, 3))
    all_vaudio = all_vaudio.reshape((200, 199, 257, 3))

    for _ in range(800):
        pair_labels.append(1)

    pair_vlabels = []

    for _ in range(100):
        pair_vlabels.append(0)
    for _ in range(100):
        pair_vlabels.append(1)


    pairs = [all_images, all_audio]
    v_pairs = [all_v_images, all_vaudio]

    #pair_labels = utils.to_categorical(pair_labels, 2)
    #pair_vlabels = utils.to_categorical(pair_vlabels, 2)

    pair_labels = np.array(pair_labels)
    pair_vlabels = np.array(pair_vlabels)

    print(pairs[0].shape, pair_labels.shape)
    return pairs, pair_labels, v_pairs, pair_vlabels


def getAudioData(dir1 = '.\\data\\audio\\airplanes\\' , dir2 = '.\\data\\audio\\motorbikes\\'):
    data = []
    labels = []

    data, labels = readImages(dir1, 0, 400, data, labels, '.jpg')
    data, labels = readImages(dir2, 1, 400, data, labels, '.jpg')

    train_data = data[:800]
    train_label = labels[:800]

    test_data = data[700:800]
    test_label = labels[700:800]

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

    test_data = data[700:800]
    test_label = labels[700:800]

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
