# import <
import numpy as np
from cv2 import imread
import tensorflow as tf
from os import listdir, path
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# >


# global <
gRealpath = path.realpath(__file__)
gData = {'star' : 500, 'galaxy' : 500}
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
    inputShape = x[0].shape
    labelEncoder = LabelEncoder()

    # >

    # one hot encode y <
    # categorize y <
    # stack x then normalize <
    y = labelEncoder.fit_transform(y)
    y = to_categorical(y, len(pData))
    x = np.stack(x, axis = 0) / 255.0

    # >

    return (x, y, inputShape)


def graphHistory(pHistory: dict, pType: str) -> None:
    '''  '''

    plt.switch_backend('TkAgg')
    plt.plot(pHistory[pType], label = 'Train')
    plt.plot(pHistory[f"val_{pType}"], label = 'Validation')
    plt.title(pType)
    plt.ylabel(f"Decimal {pType}")
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# main <
if (__name__ == '__main__'):

    # load images from data directory <
    # one hot encode and categorize y, stack x <
    x, y = zip(*loadData(pKey = gData.keys(), pValue = gData.values()))
    x, y, inputShape = translateData(pData = gData, x = x, y = y)

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

    # initialize model <
    model = Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = inputShape))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2), padding = 'same'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1024, activation = 'relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(2, activation = 'softmax'))

    # >

    # compile model <
    model.compile(

        metrics = ['accuracy'],
        loss = 'categorical_crossentropy',
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    )

    # >

    # set callbacks <
    modelCallbacks = [

        callbacks.ModelCheckpoint(filepath = '/tmp/checkpoint'),
        callbacks.ReduceLROnPlateau(monitor = 'accuracy', patience = 3, factor = 0.2),
        callbacks.EarlyStopping(monitor = 'accuracy', restore_best_weights = True, patience = 10)

    ]

    # >

    # fit model <
    history = model.fit(

        xTrain,
        yTrain,
        epochs = 50,
        shuffle = True,
        batch_size = 64,
        callbacks = modelCallbacks,
        validation_data = (xValid, yValid)

    )

    # >

    # graph accuracy <
    # graph loss <
    graphHistory(history.history, 'accuracy')
    graphHistory(history.history, 'loss')

    # >

# >