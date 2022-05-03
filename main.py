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
gData = {0 : 'star', 1 : 'galaxy'}
gRealpath = path.realpath(__file__)
gDirectory = ('/'.join(gRealpath.split('/')[:-1]))

# >


def loadData(pData: list, pMax: int = 10000) -> list:
    '''  '''

    # local <
    rData, pData = [], [f'{gDirectory}/data/{i}' for i in pData]
    f = lambda i : [rData.append((imread(f'{i}/{j}'), i.split('/')[-1])) for j in listdir(i)[:pMax]]

    # >

    # get data <
    # return data <
    [f(i) for i in pData]
    return rData

    # >


def translateData(pData: list, x: tuple, y: tuple) -> list:
    '''  '''

    # local <
    labelEncoder = LabelEncoder()

    # >

    # one hot encode y <
    # categorize y <
    y = labelEncoder.fit_transform(y)
    y = to_categorical(y, len(pData))

    # >

    # stack x <
    x = np.stack(x, axis = 0) / 255.0

    # >

    return x, y


# main <
if (__name__ == '__main__'):

    # load images from data directory <
    # one hot encode and categorize y, stack x <
    x, y = zip(*loadData(pData = gData.values(), pMax = 5))
    x, y = translateData(pData = gData.values(), x = x, y = y)

    # >

    # get train and test batch <
    # split test into test and validation batch <
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

# >