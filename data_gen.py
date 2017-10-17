from cv2 import imread
import numpy as np
import cv2
from random import shuffle

#Supposing that the images are stored like this
# imagen1
# audion1
# imgaey1
# audioy1

#normalize

path = 'path\\to\\files\\'


def getImageFromPair(id):
    image_path = path + 'paired\\image\\' + str(id)
    img = imread(image_path)
    return img

def getAudioFromPair(id):
    audio_path = path + 'paired\\audio\\' + str(id)
    aud = imread(audio_path)
    return aud




def getImageFromNotPair(id):
    image_path = path + 'not_paired\\image\\' + str(id)
    img = imread(image_path)
    return img

def getAudioFromNotPair(id):
    audio_path = path + 'not_paired\\audio\\' + str(id)
    aud = imread(audio_path)
    return aud



def readNonPairs(id):
    label = 0

    image = getImageFromNotPair(id)
    audio = getAudioFromNotPair(id)

    pair = [image, audio]
    return (pair, label)

def readPairs(id):
    label = 1
    image = getImageFromPair(id)
    audio = getAudioFromPair(id)

    pair = [image, audio]
    return (pair, label)


def getData(start, batch_size):

    posData = []
    negData = []
    for i in range(batch_size):
        posData.append(readPairs(start + i))
        negData.append(readNonPairs(start + i))

        data = posData + negData
        shuffle(data)

    return data

def getGenerator():
    return
