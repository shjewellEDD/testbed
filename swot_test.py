'''
TODO:
    Add config file, it should contain=
        Constant dicts and lists
    Analysis page
        Select set, get stats + histogram
        Select point, get all other points from
    Hoverdata for the map
    Use Tracy's email to add correct headers and footers

'''

import dash
import dash_auth
from dash import html as dhtml
from dash import dcc, dash_table
from dash.dash_table.Format import Format, Scheme
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
# import plotly.graph_objects as go
import dash_bootstrap_components as dbc

#non-plotly imports
import data_import
import datetime
import pandas as pd
import configparser
import numpy as np

# reads username password pairs from config file
config = configparser.ConfigParser()
config.read('swot_config.ini')
access_keys = {i[0]: i[1] for i in list(config['access_keys'].items())}

# reads in prawler information from a spreadsheet and generates the requisite data structures for Plotly to read
metadata = pd.read_csv(f'prawlers.csv', index_col='ID')

url_dict = dict(zip(metadata.index, metadata['url']))
prawler = [{'label': name, 'value': name} for name in metadata['prawler'].unique()]
sets = list(metadata['prawler'].unique())

subset = dict()
metanp = metadata.to_numpy()

for prawl in metadata['prawler'].unique():

    ids = metadata[metadata['prawler'] == prawl].index.to_numpy()
    names = metadata[metadata['prawler'] == prawl]['name'].to_numpy()
    temp = []

    for n, id in np.ndenumerate(ids):
        temp.append({'label': names[n], 'value': id})

    subset[prawl] = temp

set_reverse = dict()
metanp = metadata.to_numpy()

for prawl in metadata.index:
    set_reverse[prawl] = {'prawler': metadata.loc[prawl]['prawler'], 'name': metadata.loc[prawl]['name']}

lat_lons = dict()

for key, url in url_dict.items():

    dset = data_import.Dataset(url)
    meta = dset.gen_metadata()

    lat_lons[key] = {'lat': (float(meta['latitude']['max']) + float(meta['latitude']['min'])) / 2,
                     'lon': (float(meta['longitude']['max']) + float(meta['longitude']['min'])) / 2,
                     'pid': set_reverse[key]['prawler'],
                     'name': set_reverse[key]['name']}

lat_lon_df = pd.DataFrame(lat_lons).transpose()
lat_lon_df['size'] = [1]*len(lat_lon_df)

map_columns = [
    dict(id='pid', name='Prawler ID'),
    dict(id='name', name='Set Name'),
    dict(id='lat', name='Latitude', type='numeric', format=Format(precision=5, scheme=Scheme.decimal)),
    dict(id='lon', name='Longitude', type='numeric', format=Format(precision=6, scheme=Scheme.decimal))
]

graph_config = {'modeBarButtonsToRemove' : ['hoverCompareCartesian', 'select2d', 'lasso2d'],
                'doubleClick':  'reset+autosize', 'toImageButtonOptions': { 'height': None, 'width': None, },
                'displaylogo': False}

colors = {'background': '#111111', 'text': '#7FDBFF'}

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                requests_pathname_prefix='/swot/test/',
                external_stylesheets=[dbc.themes.SLATE])
server = app.server

auth = dash_auth.BasicAuth(app, access_keys)

external_stylesheets = ['https://codepen.io./chriddyp/pen/bWLwgP.css']

tools_card = dbc.Card([
    dbc.CardBody(
        dash_table.DataTable(id='map-table',
                             data=lat_lon_df.to_dict('records'),
                             columns=map_columns,
                             row_selectable='single',
                             cell_selectable=False,
                             style_table={'backgroundColor': colors['background'],
                                          'overflow'       :'auto'},
                             style_cell={'backgroundColor': colors['background'],
                                         'textColor':       colors['text']}
                             )
    )]
)

set_card = dbc.Card([
        dbc.CardBody(
            children=[
                dbc.Row(children=[
                    dbc.Col(children=[
                        dhtml.H5('Pralwer'),
                        dcc.Dropdown(
                            id='select_prawl',
                            options=prawler,
                            value=prawler[0]['value'],
                            clearable=False
                        )
                    ]),
                    dbc.Col(children=[
                        dhtml.H5('Plot'),
                        dcc.Dropdown(
                            id='select_set',
                            options=subset[sets[0]],
                            value=subset[sets[0]][0]['value'],
                            clearable=False
                        ),
                        dcc.Dropdown(
                            id="select_var",
                            clearable=False
                        )
                    ])
                ])
            ]
        )
])

overlay_card = dbc.Card([
        dbc.CardBody(
            children=[
                dhtml.H5('Overlay'),
                dcc.Dropdown(
                    id='overlay_set',
                    options=subset[sets[0]],
                    clearable=True
                ),
                dcc.Dropdown(
                    id="overlay_var",
                    clearable=True
                )
            ]
        )
])

date_card = dbc.Card([
    dbc.CardBody(
        children=[
            dhtml.H5('Date Range'),
            dcc.DatePickerRange(
                id='date-picker',
                style={'backgroundColor': colors['background']},

            ),
        ]
    )
])

table_card = dbc.Card([
    dbc.CardBody(
        children=[dcc.Textarea(id='t_mean',
                               value='',
                               readOnly=True,
                               style={'width': '100%', 'height': 40,
                                      'textColor':       colors['text']},
                               ),
                  dash_table.DataTable(id='dtable',
                                       style_table={'backgroundColor': colors['background'],
                                                    'height'         :'380px',
                                                    'overflowY'       :'auto'},
                                       style_cell={'backgroundColor': colors['background'],
                                                   'textColor':       colors['text']}
                  )
        ])
])

graph_card = dbc.Card(
    [dbc.CardBody([
        dcc.Loading(
            dcc.Graph(id='graph')
        )
    ])]
)

prawl_map = go.Figure(data=px.scatter_geo(
    data_frame=lat_lon_df,
    lat=lat_lon_df['lat'],
    lon=lat_lon_df['lon'],
    size=lat_lon_df['size'],
    size_max=5,
    opacity=1,
    hover_data={'lat': False, 'lon': False},
#     hover_data={'size': False},
#    color=df['type'],
#    color_discrete_sequence=px.colors.qualitative.D3
))


prawl_map.update_layout(
    #autosize=True,
    width=1200,
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
    )
)

map_card = dbc.Card(
    [dbc.CardBody([dcc.Graph(figure=prawl_map)]
    )]
)

load_tab = dcc.Tab(label='Load', value='load',
        children=[
            dbc.Card(
                dbc.CardBody(children=[
                    dbc.Row(
                        children=[
                            dbc.Col(dhtml.Div([date_card])),
                            dbc.Col(dhtml.Div([set_card])),
                            dbc.Col(dhtml.Div([overlay_card]))
                        ]
                    ),
                    dbc.Row(children=[
                        dbc.Col(graph_card, width=8),
                        dbc.Col(table_card, width=4)
                    ])
                ])
            )
        ])

map_tab = dcc.Tab(label='Map', value='prawl-map',
    children=[
        dbc.Card(
            dbc.CardBody(
                children=[
                dbc.Row([
                    dbc.Col(tools_card, width=4),
                    dbc.Col(map_card, width=8)
                ])
                ]
            )
        ),
    ])

header = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(width=1, children=
            [
                dhtml.Img(src='noaa-logo-rgb-2022.png', height=100,
                          width=100)
            ]),
            dbc.Col(width=11, children=[
                dhtml.H1('SWOT Prawlers')
            ])
        ])
    ])
)

footer = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(width=1, children=
            [
                dhtml.Img(src='https://www.pmel.noaa.gov/sites/default/files/PMEL-meatball-logo-sm.png', height=100,
                         width=100)
            ]),
            dbc.Col(width=11, children=[
                dhtml.Div(children=[
                    dcc.Link('National Oceanic and Atmospheric Administration', href='https://www.noaa.gov/'),
                ]),
                dhtml.Div(children=[
                    dcc.Link('Pacific Marine Environmental Laboratory', href='https://www.pmel.noaa.gov/'),
                ]),
                dhtml.Div(children=[
                    dcc.Link('oar.pmel.webmaster@noaa.gov', href='mailto:oar.pmel.webmaster@noaa.gov')
                ]),
                dhtml.Div(children=[
                    dcc.Link('DOC |', href='https://www.commerce.gov/'),
                    dcc.Link(' NOAA |', href='https://www.noaa.gov/'),
                    dcc.Link(' OAR |', href='https://www.research.noaa.gov/'),
                    dcc.Link(' PMEL |', href='https://www.pmel.noaa.gov/'),
                    dcc.Link(' Privacy Policy |', href='https://www.noaa.gov/disclaimer'),
                    dcc.Link(' Disclaimer |', href='https://www.noaa.gov/disclaimer'),
                    dcc.Link(' Accessibility', href='https://www.pmel.noaa.gov/accessibility')
                ])
            ])
        ])
    ])
)

app.layout = dhtml.Div([
    header,
    dbc.Card(
        dbc.CardBody(
    #dbc.Container([(
        dcc.Tabs(id='selected-tab', value='prawl-map',
                 children=[
                     map_tab,
                     load_tab
                ])
        )),
    footer
])


'''
========================================================================================================================
Callbacks
'''

#prawler selection
@app.callback(
    [Output('select_set',   'options'),
     Output('select_prawl', 'value'),
     Output('selected-tab', 'value'),
     Output('overlay_set',  'options')],
    [Input('select_prawl',  'value'),
     Input('map-table',     'selected_rows')
     ]
)

def select_prawler(drop_val, table_val):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if (changed_id == '.') or (changed_id == 'map-table.derived_virtual_row_ids'):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    elif (changed_id == 'map-table.selected_rows') and (table_val == [None]):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    elif (changed_id == 'map-table.selected_rows'):
        drop_val = lat_lon_df.iloc[table_val[0]]['pid']

    return subset[drop_val], drop_val, 'load', subset[drop_val]

#engineering data selection
@app.callback(
    [Output('select_var', 'options'),
    Output('date-picker', 'min_date_allowed'),
    Output('date-picker', 'max_date_allowed'),
    Output('date-picker', 'start_date'),
    Output('date-picker', 'end_date'),
    Output('select_var',  'value')],
    Input('select_set',   'value'))

def change_set(dataset):
    '''
    Updates dropdowns and date ranges when prawler dataset is changed.
    :param dataset:
    :return:
    '''

    eng_set = data_import.Dataset(url_dict[dataset])

    #min_date_allowed = eng_set.t_start.date()
    #max_date_allowed = eng_set.t_end.date()
    min_date_allowed = eng_set.data_start()
    max_date_allowed = eng_set.data_end()
    start_date = (max_date_allowed - datetime.timedelta(days=14))
    end_date = max_date_allowed
    first_var = eng_set.ret_vars()[0]

    return eng_set.gen_drop_vars(), min_date_allowed, max_date_allowed, start_date, end_date, first_var


#overlay selection
@app.callback(
    Output('overlay_var', 'options'),
    Input('overlay_set', 'value')
)

def overlay_vars(prawl):
    '''
    Populates overlay variables dropdown

    :param prawl:
    :return:
    '''

    if not prawl:

        return []

    dset = data_import.Dataset(url_dict[prawl])

    return dset.gen_drop_vars()

#engineering data selection
@app.callback(
    [Output('graph', 'figure'),
     Output('dtable', 'data'),
     Output('dtable', 'columns'),
     Output('t_mean', 'value')],
    [Input('select_set', 'value'),
     Input('select_var', 'value'),
     Input('overlay_var', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')],
     State('overlay_set', 'value'))


def plot_evar(dataset, select_var, ovr_var, start_date, end_date, ovr_prawl):
    '''
    Because plotly only allows an graphical object be touched by a single callback, everything that changes the graph
    is contain here, making this a monstrosity.

    The general structure is:
        load data
        if-then block if the user wants a custom variable (x-per-day)
        layout changes

    :param dataset:
    :param select_var:
    :param ovr_var:
    :param start_date:
    :param end_date:
    :param ovr_prawl:
    :return:
    '''

    eng_set = data_import.Dataset(url_dict[dataset])
    new_data = eng_set.get_data(window_start=start_date, window_end=end_date, variables=[select_var])

    #changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if ovr_var:

        ovr_set = data_import.Dataset(url_dict[ovr_prawl])
        ovr_data = ovr_set.get_data(window_start=start_date, window_end=end_date, variables=[ovr_var])

        ovr_data.index = ovr_data['time']
        new_data.index = new_data['time']

        ovr_data = pd.concat(map(lambda c: ovr_data[c].dropna().reindex(new_data['time'], method='nearest'),
                                 ovr_data.columns), axis=1)

        new_data[ovr_var] = ovr_data[ovr_var]

    colorscale = px.colors.sequential.Viridis

    if select_var == 'trips_per_day':
        trip_set = eng_set.trips_per_day(start_date, end_date)
        efig = px.scatter(trip_set, y='ntrips', x='days')#, color="sepal_length", color_continuous_scale=colorscale)

        columns = [{"name": 'Day', "id": 'days'},
                   {'name': select_var, 'id': 'ntrips'}]

        t_mean = "Mean Trips per day: " + str(round(trip_set['ntrips'].mean(), 3))

        try:
            table_data = trip_set.to_dict('records')
        except TypeError:
            table_data = trip_set.to_dict()

    elif select_var == 'errs_per_day':

        err_set = eng_set.errs_per_day(start_date, end_date)
        efig = px.scatter(err_set, y='nerrors', x='days')#, color="sepal_length", color_continuous_scale=colorscale)

        columns = [{"name": 'Day', "id": 'days'},
                   {'name': select_var, 'id': 'nerrors'}]

        t_mean = 'Mean errors per day ' + str(round(err_set['nerrors'].mean(), 3))

        try:
            table_data = err_set.to_dict('records')
        except TypeError:
            table_data = err_set.to_dict()

    elif select_var == 'sci_profs':

        sci_set = eng_set.sci_profiles_per_day(start_date, end_date)
        efig = px.scatter(sci_set, y='ntrips', x='days')#, color="sepal_length", color_continuous_scale=colorscale)

        columns = [{"name": 'Day', "id": 'days'},
                   {'name': select_var, 'id': 'ntrips'}]

        t_mean = 'Mean errors per day ' + str(round(sci_set['ntrips'].mean(), 3))

        try:
            table_data = sci_set.to_dict('records')
        except TypeError:
            table_data = sci_set.to_dict()

    else:
        if ovr_var:
            efig = px.scatter(y=new_data[select_var], x=new_data['time'], color=ovr_data[ovr_var],
                              color_continuous_scale=colorscale)
            efig.layout['coloraxis']['colorbar']['title']['text'] = ovr_var

        else:
            efig = px.scatter(new_data, y=select_var, x='time')

        columns = [{"name": 'Date', "id": 'timestring'},
                   {'name': select_var, 'id': select_var}]

        if ovr_var:
            columns.append({'name': ovr_var, 'id': ovr_var})

        try:
            t_mean = f'Average {select_var}: {str(round(new_data.loc[:, select_var].mean(), 3))}'
        except TypeError:
            t_mean = ''

        try:
            table_data = new_data.to_dict('records')
        except TypeError:
            table_data = new_data.to_dict()

    if 'depth' in select_var.lower():

        efig['layout']['yaxis']['autorange'] = "reversed"

    efig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        yaxis_title=select_var,
    )

    return efig, table_data, columns, t_mean


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)
    app.run_server(debug=True)