# import <
from dash import html, dcc
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

            # <
            # <
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
                    color = gBlackColor,
                    gFontFamily = gFontFamily

                )

            )

        ]

    )


def graphFunction():
    '''  '''

    pass


def predictFunction():
    '''  '''

    pass


def frontendFunction():
    '''  '''

    return dbc.Container(

        fluid = True,
        style = dict(

            top = 0,
            left = 0,
            width = '100%',
            height = '100%',
            position = 'fixed',
            backgroundColor = gBlackColor

        ),
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
                children = graphFunction()

            ),
            dbc.Row(

                justify = 'center',
                children = predictFunction()

            )

            # >

        ]

    )