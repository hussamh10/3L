import numpy as np
import cv2

size = 200

def getWhiteAudio():
    #img = cv2.imread('data\\audio\\white')
    img = np.zeros((199, 257, 3))
    img = img+255
    img /= 255
    return img

def getBlackAudio():
    #img = cv2.imread('data\\audio\\black')
    img = np.zeros((199, 257, 3))
    img /= 255
    return img

def getWhiteVideo_old(i):
    img = np.zeros((224, 224, 3)) + 255
    img /= 255
    return img

    
def getBlackVideo_old(i):
    img = np.zeros((224, 224, 3))
    img /= 255
    return img


def getWhiteVideo(i):
    i = (i%10) + 1
    name = 'image (' + str(i) + ').jpg'
    img = cv2.imread('data\\video\\white\\' + name)
    img = img.astype('float32')
    #img = np.zeros((224, 224, 3))
    #img = img+255
    img /= 255
    return img

def getBlackVideo(i):
    i = (i%10) + 1
    name = 'image (' + str(i) + ').jpg'
    img = cv2.imread('data\\video\\black\\' + name)
    img = img.astype('float32')
    #img = np.zeros((224, 224, 3))
    img /= 255
    return img

def getFusionData():
    v1 = []
    v2 = []
    for i in range(size):
        v1.append(getBlackVideo_old(i))
    for i in range(size):
        v2.append(getWhiteVideo_old(i))

    a1 = []
    a2 = []
    for i in range(size):
        a1.append(getBlackAudio())
    for i in range(size):
        a2.append(getWhiteAudio())

    # v1 = black
    # v2 = white
    # a1 = black
    # a2 = white
    v1 = np.array(v1)
    v2 = np.array(v2)

    a1 = np.array(a1)
    a2 = np.array(a2)

    print(v1.shape)

    V12 = np.append(v1, v2, axis=0)
    A12 = np.append(a1, a2, axis=0)


    dupV12 = np.append(v1, v2, axis=0)
    A21 = np.append(a2, a1, axis=0)

    V12 = np.array(V12)
    A12 = np.array(A12)

    dupV12 = np.array(dupV12)
    A21 = np.array(A21)

    print(A21.shape)

    all_V = np.append(V12, dupV12, axis=0)
    all_A = np.append(A12, A21, axis=0)

    labels_p = np.zeros(size*2) + 1
    labels_n = np.zeros(size*2)

    labels = np.append(labels_p, labels_n)

    return [all_V, all_A], labels
