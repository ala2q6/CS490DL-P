# import <
import numpy as np
import pandas as pd
from dash import html, dcc
from plotly import express as pe
import dash_bootstrap_components as dbc

# >


# global <
gBlackColor = '#111111'
gWhiteColor = '#F8F0E3'
gFontFamily = 'sans-serif'

# >


def headerFunction():
    '''  '''

    return dbc.Col(

        style = dict(

            marginTop = 10,
            marginLeft = 10,
            marginRight = 35,
            borderRadius = 10,
            textAlign = 'center',
            backgroundColor = gWhiteColor

        ),
        children = [

            # title <
            # subtitle <
            # divider <
            html.H1(

                children = 'Galaxy Star Classification',
                style = dict(

                    margin = 0,
                    color = gBlackColor,
                    fontFamily = gFontFamily

                )

            ),
            html.P(

                children = 'by JA2',
                style = dict(

                    margin = 0,
                    fontSize = 20,
                    color = gBlackColor,
                    fontFamily = gFontFamily

                )

            ),
            html.Hr(style = dict(border = f'1px solid {gBlackColor}')),

            # >

            # model summary <
            # divider <
            html.Img(

                src = 'https://bit.ly/3sigGGP',
                style = {

                    "width" : "100%",
                    "display" : "block",
                    "borderRadius" : 10,
                    "position" : "center",
                    "marginLeft" : "auto",
                    "marginRight" : "auto"

                }

            ),
            html.Hr(style = dict(border = f'1px solid {gBlackColor}'))

            # >

        ]

    )


def graphFunction(history):
    ''' graph training and validation
        and accuracy, loss, auc; three graphs'''

    # local <
    f = lambda pHistory, pType : pe.line(

        title = pType,
        y = ['train', 'validation'],
        data_frame = pd.DataFrame({

            'train' : pHistory[pType],
            'validation' : pHistory[f'val_{pType}']

        })

    )

    # >

    return dbc.Col(

        style = dict(

            marginTop = 10,
            marginLeft = 10,
            marginRight = 35,
            borderRadius = 10,
            textAlign = 'center',
            backgroundColor = gWhiteColor

        ),
        children = [

            dcc.Graph(figure = f(history.history, 'accuracy')),
            dcc.Graph(figure = f(history.history, 'loss')),
            dcc.Graph(figure = f(history.history, 'auc'))

        ]

    )


def predictFunction():
    '''  '''

    return dbc.Col(

        style = dict(

            marginTop = 10,
            marginLeft = 10,
            marginRight = 35,
            borderRadius = 10,
            textAlign = 'center',
            backgroundColor = gWhiteColor

        ),
        children = [

            dcc.Upload(

                id = 'uploadId',
                children = [

                    'Drag and Drop or',
                    html.A('Select a File')

                ],
                style = dict(

                    marginTop = '5%',
                    padding = '5%',
                    marginBottom = '5%',
                    borderRadius = 10,
                    border = 'dashed'

                )

            ),
            html.H1(id = 'outputId')

        ]

    )


def frontendFunction(history):
    '''  '''

    return dbc.Container(

        fluid = True,
        style = {

            'top' : 0,
            'left' : 0,
            'width' : '100%',
            'height' : '100%',
            'position' : 'fixed',
            'overflow-x' : "hidden",
            'overflow-y' : "scroll",
            'backgroundColor' : gBlackColor

        },
        children = [

            # <
            # <
            # <
            dbc.Row(

                justify = 'center',
                children = headerFunction()

            ),
            dbc.Row(

                justify = 'center',
                children = graphFunction(history)

            ),
            dbc.Row(

                justify = 'center',
                children = predictFunction()

            )

            # >

        ]

    )
