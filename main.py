# import <
import numpy as np
from cv2 import imread
import tensorflow as tf
from os import listdir, path

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# >


# global <
gRealpath = path.realpath(__file__)
gData = {'star' : 50, 'galaxy' : 50}
gDirectory = ('/'.join(gRealpath.split('/')[:-1]))

# >


def loadData(pKey: list, pValue: list) -> list:
    '''  '''

    # local <
    rData, pKey = [], [f'{gDirectory}/data/{i}' for i in pKey]
    f = lambda k, v : [rData.append((imread(f'{k}/{i}'), k.split('/')[-1])) for i in listdir(k)[:v]]

    # >

    # get data <
    # return data <
    [f(k, v) for k, v in zip(pKey, pValue)]
    return rData

    # >


def translateData(pData: list, x: tuple, y: tuple) -> tuple:
    '''  '''

    # local <
    labelEncoder = LabelEncoder()

    # >

    # one hot encode y <
    # categorize y <
    y = labelEncoder.fit_transform(y)
    y = to_categorical(y, len(pData))

    # >

    # stack x and normalize <
    x = np.stack(x, axis = 0) / 255.0

    # >

    return (x, y)


# main <
if (__name__ == '__main__'):

    # load images from data directory <
    # one hot encode and categorize y, stack x <
    x, y = zip(*loadData(pKey = gData.keys(), pValue = gData.values()))
    x, y = translateData(pData = gData, x = x, y = y)

    # >

    # # # get train and test batch <
    # # # split test into test and validation batch <
    xTrain, xTest, yTrain, yTest = train_test_split(

        x,
        y,
        shuffle = True,
        test_size = 0.3,
        random_state = 42

    )
    xTest, xValid, yTest, yValid = train_test_split(

        xTest,
        yTest,
        shuffle = True,
        test_size = 0.5,
        random_state = 42

    )

    # >

    # <


    # >

# >