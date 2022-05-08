# import <
import numpy as np
from cv2 import imread
from multiprocessing import Pool
from os import listdir, path, mkdir
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
gRealpath = path.realpath(__file__)
gData = {'star' : 5000, 'galaxy' : 5000}
gDirectory = ('/'.join(gRealpath.split('/')[:-1]))

# >


def loadData(pFile: str, pKey: list, pValue: list) -> list:
    '''  '''

    # local <
    rData = []
    f = lambda k, v, l : [rData.append((imread(f'{k}/{i}'), l)) for i in listdir(k)[:v]]

    # >

    # get aData <
    # return aData <
    [f(k = f'{gDirectory}/{pFile}/{k}', v = v, l = k) for k, v in zip(pKey, pValue)]
    return rData

    # >


def filterImage(pImg: np.ndarray, pType: str, pFile: str, i: str) -> None:
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
    rInputShape = x[0].shape
    lDir = f'{gDirectory}/bData'
    labelEncoder = LabelEncoder()

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
                pFile,
                str(i)

            ) for i, img in enumerate(x)]

        )

        # >

    # >


def translateData(x: tuple, y: tuple) -> tuple:
    '''  '''

    # local <
    inputShape = x[0].shape
    labelEncoder = LabelEncoder()

    # >

    # one hot encode y <
    # categorize y <
    # stack x then normalize <
    y = labelEncoder.fit_transform(y)
    y = to_categorical(y, len(gData))
    x = np.stack(x, axis = 0) / 255.0

    # >

    return (x, y, inputShape)


def featureExtraction(model):
    '''  '''

    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:
          continue
        print(i, layer.name, layer.output.shape)

    # create model to ouptut right after the first hidden layer
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


def displayImages(data, labels = None) -> None:
    '''  '''

    plt.figure(figsize=(8,8))

    for i in range(16):
        plt.subplot(4,4,i+1)
        if labels is not None:
            plt.title(y[i])
        plt.tick_params(
            left = False,
            labelleft = False,
            bottom = False,
            labelbottom = False
        )
        plt.imshow(data[i])

    plt.show()


# main <
if (__name__ == '__main__'):

    # load aData <
    # optional filer <
    # categorize y and stack x <
    print('Loading Data...')
    x, y = zip(*loadData(

        pFile = 'aData',
        pKey = gData.keys(),
        pValue = gData.values()

    ))
    # print('Filter Running...')
    # filterData(
    #
    #     x = x,
    #     pProcessor = 4,
    #     pFile = 'bData'
    #
    # )
    print('Translating Data...')
    x, y, inputShape = translateData(

        x = x,
        y = y

    )

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
        test_size = 0.5,
        random_state = 42

    )

    # >

    # local <
    dropFactor = 0.15
    # filSize = (3, 3)

    # >

    # initialize model <
    print('Building Model...')
    model = Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = inputShape))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (5, 5), activation = 'relu'))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Conv2D(1024, (5, 5), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2), padding = 'same'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Dense(1024, activation = 'relu'))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(dropFactor))

    model.add(layers.Dense(2, activation = 'softmax'))

    # >

    model.summary()

    # compile model <
    model.compile(

        metrics = ['accuracy'],
        loss = 'categorical_crossentropy',
        optimizer = optimizers.SGD(learning_rate = 0.01)

    )

    # >

    # set callbacks <
    modelCallbacks = [

        callbacks.ModelCheckpoint(filepath = '/tmp/checkpoint'),
        callbacks.ReduceLROnPlateau(monitor = 'accuracy', patience = 5, factor = 0.2),
        callbacks.EarlyStopping(monitor = 'accuracy', restore_best_weights = True, patience = 10)

    ]

    # >

    # initialize parameters for image augmentation <
    augmentor = ImageDataGenerator(

        shear_range = 20,
        vertical_flip = True,
        rotation_range = 0.25,
        horizontal_flip = True,
        zoom_range = [0.5, 1.5],
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        brightness_range = [.75, 1.25]

    )

    # >

    # fit generator to train aData <
    trainGen = augmentor.flow(

        xTrain,
        yTrain,
        batch_size = 16

    )

    # >

    # fit model with image generator <
    history = model.fit_generator(

        trainGen,
        epochs = 25,
        shuffle = True,
        steps_per_epoch = 50,
        callbacks = modelCallbacks,
        validation_data = (xValid, yValid)

    )

    # >

    # limit output precision for floats <
    with np.printoptions(precision=4):

        # local <
        prediction = model.predict(xTest)
        scores = model.evaluate(xTest, yTest, verbose = 0)

        # >

        # test model on unused aData <
        # use model to classify unused aData <
        print('\n\ntest loss = ', scores[0], '\ntest accuracy = ', scores[1])
        for i in range(0, 10): print('\nTrue = ', yTest[i], '\nPred = ', prediction[i])

        # >

    # >

    # graph accuracy <
    # graph loss <
    # graphHistory(history.history, 'accuracy')
    # graphHistory(history.history, 'loss')

    # >

    # model.save(f'{directory}/model/{gVersion}')

# >