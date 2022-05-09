# import <
from os import path
from dash import Dash
from multiprocessing import Process # remove
import dash_bootstrap_components as dbc

from frontend import frontendFunction
from backend import buildModel, trainModel
from backend import loadData, translateData

# >


# global <
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

    # <
    # <
    application.layout = frontendFunction(

        #

    )
    application.run_server()

    # >

# >