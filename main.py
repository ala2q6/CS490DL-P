# import <
import numpy as np
from os import path
from dash import Dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from frontend import frontendFunction
from backend import buildModel, trainModel
from backend import loadData, translateData

# >


# global <
gModel = None
gRealpath = path.realpath(__file__)
gData = {'star' : 5000, 'galaxy' : 5000}
gDirectory = ('/'.join(gRealpath.split('/')[:-1]))
application = Dash(

    suppress_callback_exceptions = True,
    external_stylesheets = [dbc.themes.GRID]

)
server = application.server

# >


# main <
if (__name__ == '__main__'):

    # <
    # <
    x, y = zip(*loadData(

        pFile = 'aData',
        pDir = gDirectory,
        pKey = gData.keys(),
        pValue = gData.values()

    ))
    x, y, inputShape = translateData(

        x = x,
        y = y,
        pData = gData

    )

    # >

    # <
    # <
    model = buildModel(

        inputShape = inputShape

    )

    history, model = trainModel(

        x = x,
        y = y,
        model = model

    )

    # >

    # test model on unused aData <
    # use model to classify unused aData <
    scores = model.evaluate(x[1], y[1], verbose = 0)
    prediction = model.predict(x[1])

    # limit output precision for floats <
    with np.printoptions(precision = 4):

        # output test evaluation of model <
        # output test classifications from model <
        print('\nModel Evaluation\ntest loss = ', scores[0], '\ntest accuracy = ', scores[1], '\ntest AUC = ', scores[2])
        print('\nModel Prediction')
        for i in range(0, 10): print('True = ', y[1][i], '\nPred = ', prediction[i], '\n')

        # >

    # >

    # <
    # <
    application.layout = frontendFunction(

        history = history

    )
    application.run_server()

    # >

# >


# @application.callback(
#
#     Input('uploadId', 'children'),
#     Output('outputId', 'children')
#
# )
# def callbackFunction(*args):
#     '''  '''
#
#     print('ok')
#     return None
