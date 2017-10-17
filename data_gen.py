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

path = 'data\\'


def getImageFromPair(id):
    image_path = path + 'pair\\images\\' + str(id) + '.jpg'
    img = imread(image_path)
    return img

def getAudioFromPair(id):
    audio_path = path + 'pair\\audio\\' + str(id) + '.jpg'
    aud = imread(audio_path)
    return aud




def getImageFromNotPair(id):
    image_path = path + 'non_pair\\images\\' + str(id) + '.jpg'
    img = imread(image_path)
    return img

def getAudioFromNotPair(id):
    audio_path = path + 'non_pair\\audio\\' + str(id) + '.jpg'
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


def getData(batches, batch_size):
    start = 1

    for batch_no in range(batches):

        start = batch_no*batch_size
        start = start+1
        all_audio = []
        all_images = []
        posData = []
        negData = []
        labels = []
        pairs = []
        for i in range((batch_size)):
            id = start + i
            posData.append(readPairs(start + i))

            negData.append(readNonPairs(start + i))

            tpls = posData + negData
            shuffle(tpls)

        for tpl in tpls:
            labels.append(tpl[1])
            pairs.append(tpl[0])

        for pair in pairs:
            all_images.append(pair[0])
            all_audio.append(pair[1])

        labels = np.array(labels)

        all_audio = np.array(all_audio)
        all_images = np.array(all_images)

        all_images = all_images.astype('float32')
        all_audio = all_audio.astype('float32')

        all_images /= 255
        all_audio /= 255

        pairs = [all_images, all_audio]

        yield pairs, labels

def getGenerator():
    return
