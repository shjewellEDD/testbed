import dash
from dash import html as dhtml
from dash import dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

import pandas as pd
'''
Map with various Prawlers listed on it
Visualization considerations
    Map should have locations of Prawler
    Color should indicated:
    Hoverdata should include:
        Dates covered
        Data recorded
'''

prawlers = [{'label': 'M200', 'value': 'M200'},
            {'label': 'TELON001', 'value': 'TELON001'},
            {'label': 'TELONAS2', 'value': 'TELONAS2'},
            {'label': 'ASVCO2 Real Time', 'value': 'co2rt'},
            {'label': 'ASVCO2 Validation', 'value': 'co2valid'}
            ]

data_loc = {'M200': 'https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELOM200_PRAWC_M200',
            'TELON001': 'https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELON001_PRAWE_N001',
            'TELONAS2': 'https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELONAS2_PRAWE_NAS2'
            }


gps_coords = {'M200': {'type': 'prawler', 'lat': 58,   'lon': -165},
              'TELON001': {'type': 'prawler', 'lat': 40, 'lon': -125},
              'TELONAS2': {'type': 'prawler', 'lat': 42, 'lon': -123},
              'ASVCO2 Real Time': {'type': 'co2', 'lat': 37.64, 'lon': -122.77},
              'ASVCO2 Validation': {'type': 'co2', 'lat': 47.7, 'lon': -122.26}
              }

df = pd.DataFrame(gps_coords).transpose()
df['pid'] = df.index
df['size'] = [1]*len(df)

'''
========================================================================================================================
Start Dashboard
'''

# graph_config = {'modeBarButtonsToRemove' : ['hoverCompareCartesian','select2d', 'lasso2d'],
#                 'doubleClick':  'reset+autosize', 'toImageButtonOptions': { 'height': None, 'width': None, },
#                 'displaylogo': False}

colors = {'background': '#111111', 'text': '#7FDBFF'}

external_stylesheets = ['https://codepen.io./chriddyp/pen/bWLwgP.css']

#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2015_06_30_precipitation.csv')


fig = go.Figure(data=px.scatter_geo(
    data_frame=df,
    lat=df['lat'],
    lon=df['lon'],
    size=df['size'],
    size_max=5,
    opacity=1,
    #hover_data={'type': True, 'lat': False, 'lon': False},
    hover_data={'size': False},
    color=df['type'],
    color_discrete_sequence=px.colors.qualitative.D3
))


fig.update_layout(
    #autosize=True,
    width=1200,
    #height='auto',
    margin=dict(l=5, r=5, b=5, t=5),
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    legend_title_text='',
    legend=dict(
        yanchor='top',
        y=0.99,
        xanchor='right',
        x=0.99
    ),
    geo=dict(
        showland=True,
        landcolor="slategray",
        showocean=True,
        oceancolor='darkslateblue',
        subunitcolor="slategray",
        countrycolor="slategray",
        showlakes=True,
        lakecolor="slategray",
        showsubunits=True,
        showcountries=True,
        showframe=False,
        scope='world',
        resolution=50,
        projection=dict(
            #type='conic conformal',
            type='transverse mercator'
            #rotation_lon=-100
        ),
        lonaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            range=[-200.0, -100.0],
            dtick=5
        ),
        lataxis=dict(
            showgrid=True,
            gridwidth=0.5,
            range=[35.0, 65.0],
            dtick=5
        )
    ),
)
#fig.show()

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                external_stylesheets=[dbc.themes.SLATE])
#server = app.server

tools_card = dbc.Card([
    dbc.CardBody(
        dash_table.DataTable(id='table',
                             data=df.to_dict('records'),
                             columns=[{'name':  'Set Name', 'id': 'pid'},
                                      {'name':  'Type', 'id': 'type'},
                                      {'name':  'Lat', 'id': 'lat'},
                                      {'name':  'Lon', 'id': 'lon'}],
                             style_table={'backgroundColor': colors['background'],
                                          'overflow'       :'auto'},
                             style_cell={'backgroundColor': colors['background'],
                                         'textColor':       colors['text']}
                             )
    )]
)

graph_card = dbc.Card(
    [dbc.CardBody(
         [dcc.Loading(dcc.Graph(figure=fig,
                                ))]
    )]
)

app.layout = dhtml.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([dhtml.H1('EDD Projects')]),
            dbc.Row([
                dbc.Col(tools_card, width=4),
                dbc.Col(graph_card, width=8)
            ])
        ])
    )
])

if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)