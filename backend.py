# import <
import numpy as np
from cv2 import imread
from os import listdir, mkdir
from multiprocessing import Pool
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, optimizers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# >


# global <
gVersion = '1.0'
gRGBthen = np.array([0, 0, 0])
gRGBif = np.array([195, 195, 195])

# >


def loadData(pDir: str, pFile: str, pKey: list, pValue: list) -> list:
    '''  '''

    # local <
    rData = []
    f = lambda k, v, l : [rData.append((imread(f'{k}/{i}'), l)) for i in listdir(k)[:v]]

    # >

    # get aData <
    # return aData <
    [f(k = f'{pDir}/data/{pFile}/{k}', v = v, l = k) for k, v in zip(pKey, pValue)]
    return rData

    # >


def filterImage(pImg: np.ndarray, pType: str, pDir: str, pFile: str, i: str) -> None:
    '''  '''

    # get filtered image <
    # set filtered image <
    temp = np.where(pImg < gRGBif, gRGBthen, pImg)
    plt.imshow(temp, cmap = 'gray')
    plt.savefig(f'{gDirectory}/{pFile}/{pType}/{i}.jpg')

    # >


def filterData(pFile: str, x: tuple, pProcessor: int = 4) -> None:
    '''  '''

    # local <
    lDir = f'{gDirectory}/data/{pFile}'

    # >

    # if (directory dne) then boot <
    if (not path.isdir(lDir)):

        mkdir(lDir)
        mkdir(f'{lDir}/star')
        mkdir(f'{lDir}/galaxy')

    # >

    # instantiate p processes to filter image <
    with Pool(processes = pProcessor) as pool:

        # assign function with iterable <
        pool.starmap(

            func = filterImage,
            iterable = [(

                img,
                y[i],
                lDir,
                pFile,
                str(i)

            ) for i, img in enumerate(x)]

        )

        # >

    # >


def translateData(pData: dict, x: tuple, y: tuple) -> tuple:
    '''  '''

    # local <
    inputShape = x[0].shape
    labelEncoder = LabelEncoder()

    # >

    # convert data to integers <
    # one-hot encode integers <
    # stack x then normalize <
    y = labelEncoder.fit_transform(y)
    y = to_categorical(y, len(pData))
    x = np.stack(x, axis = 0) / 255.0

    # >

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
        test_size = 0.5,
        random_state = 42

    )

    x = (xTrain, xTest, xValid)
    y = (yTrain, yTest, yValid)

    return (x, y, inputShape)


def featureExtraction(model) -> None:
    '''  '''

    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:
          continue
        print(i, layer.name, layer.output.shape)

    # create model to output right after the first hidden layer
    ixs = [3, 6, 10]
    outputs = [model.layers[i].output for i in ixs]
    disModel = Model(inputs=model.inputs, outputs=outputs)
    # get feature map for first hidden layer
    featureMaps = disModel.predict(xTest[0])
    # plot the output from each block
    square = 8
    for fmap in featureMaps:
    # plot all 64 maps in a 8x8 square
        ix = 1
        for _ in range(square):
          for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1

    # show the figure
    plt.show()


def buildModel(inputShape: tuple):
    '''  '''

    # local <
    dropFactor = 0.2

    # >

    # initialize model <
    model = Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = inputShape))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Conv2D(512, (3, 3), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding = 'same'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Dense(2, activation = 'softmax'))

    # >

    # configure model  <
    model.compile(

        metrics = ['accuracy', 'AUC'],
        loss = 'categorical_crossentropy',
        optimizer = optimizers.SGD(learning_rate = 0.01)

    )

    # >

    return model


def trainModel(x: tuple, y: tuple, model):
    '''  '''

    # set callbacks <
    modelCallbacks = [

        callbacks.ModelCheckpoint(filepath = '/tmp/checkpoint'),
        callbacks.ReduceLROnPlateau(monitor = 'accuracy', patience = 5, factor = 0.5),
        callbacks.EarlyStopping(monitor = 'accuracy', restore_best_weights = True, patience = 10)

    ]

    # >

    # initialize parameters for image augmentation <
    augmentor = ImageDataGenerator(

        shear_range = 20,
        vertical_flip = True,
        rotation_range = 0.5,
        horizontal_flip = True,
        zoom_range = [0.5, 1.5],
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        brightness_range = [0.75, 1.5]

    )

    # >

    # fit generator to train aData <
    trainGen = augmentor.flow(

        x[0],
        y[0],
        batch_size = 32

    )

    # >

    # fit model with image generator <
    history = model.fit_generator(

        trainGen,
        epochs = 25, # < insert 25
        shuffle = True,
        steps_per_epoch = 50, # < insert 50
        callbacks = modelCallbacks,
        validation_data = (x[2], y[2])

    )

    # >

    return history, model
