'''
TODO:
    Features:
        Authentication
        Handing off between dashboards OR combine into a single dashboard
    Improvement:
        Date errors:
            M200 science has a bunch of dates from the 70s and 80s
            How do we deal with this
            Just drop?
            Linearly interpolate?
    Bugs:


'''

import dash
import dash_auth
from dash import html as dhtml
from dash import dcc, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
# import plotly.graph_objects as go
import dash_bootstrap_components as dbc

#non-plotly imports
import data_import
import datetime
import pandas as pd

# PASSWORD... given that this is supposed to be more of a token barrier rather than proper security, where should we put
# this?
access_keys = {
    'pmel':    'realize'
}

prawler = [{'label': 'M200', 'value': 'M200'}]

subset = {'M200':   [{'label':   'M200 Eng', 'value': 'M200Eng'},
                    {'label':   'M200 Sci', 'value': 'M200Sci'}
                    ]
          }


'''
========================================================================================================================
Start Dashboard
'''

dataset_dict = {
            'M200Eng': 'https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELOM200_PRAWE_M200.csv',
            'M200Sci': 'https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELOM200_PRAWC_M200.csv'
            }



graph_config = {'modeBarButtonsToRemove' : ['hoverCompareCartesian','select2d', 'lasso2d'],
                'doubleClick':  'reset+autosize', 'toImageButtonOptions': { 'height': None, 'width': None, },
                'displaylogo': False}

colors = {'background': '#111111', 'text': '#7FDBFF'}

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
#                requests_pathname_prefix='/swot/test/',
                external_stylesheets=[dbc.themes.SLATE])
#server = app.server

auth = dash_auth.BasicAuth(app, access_keys)


external_stylesheets = ['https://codepen.io./chriddyp/pen/bWLwgP.css']

set_card = dbc.Card([
        dbc.CardBody(
            children=[
                dhtml.H5('Plot'),
                dcc.Dropdown(
                    id="select_eng",
                    options=subset['M200'],
                    value=subset['M200'][0]['value'],
                    clearable=False
                ),
                dcc.Dropdown(
                    id="select_var",
                    clearable=False
                )
            ]
        )
])

overlay_card = dbc.Card([
        dbc.CardBody(
            children=[
                dhtml.H5('Overlay'),
                dcc.Dropdown(
                    id="overlay_prawler",
                    options=subset['M200'],
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

app.layout = dhtml.Div([
    dbc.Container([
            dbc.Row([dhtml.H1('Prawler M200')]),
            dbc.Row([
                dbc.Col(dhtml.Div([date_card])),
                dbc.Col(dhtml.Div([set_card])),
                dbc.Col(dhtml.Div([overlay_card]))
            ]),
            dbc.Row(children=[
                dbc.Col(graph_card, width=9),
                dbc.Col(table_card, width=3)
                    ])
               ]),
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


'''
========================================================================================================================
Callbacks
'''

#engineering data selection
@app.callback(
    [Output('select_var', 'options'),
    Output('date-picker', 'min_date_allowed'),
    Output('date-picker', 'max_date_allowed'),
    Output('date-picker', 'start_date'),
    Output('date-picker', 'end_date'),
    Output('select_var', 'value')],
    Input('select_eng', 'value'))

def change_prawler(dataset):
    '''
    Updates dropdowns and date ranges when prawler dataset is changed.
    :param dataset:
    :return:
    '''

    eng_set = data_import.Dataset(dataset_dict[dataset])

    min_date_allowed = eng_set.t_start.date()
    max_date_allowed = eng_set.t_end.date()
    start_date = (eng_set.t_end - datetime.timedelta(days=14)).date()
    end_date = eng_set.t_end.date()
    first_var = eng_set.ret_vars()[0]

    return eng_set.gen_drop_vars(), min_date_allowed, max_date_allowed, start_date, end_date, first_var


#overlay selection
@app.callback(
    Output('overlay_var', 'options'),
    Input('overlay_prawler', 'value')
)

def overlay_vars(prawl):
    '''
    Populates overlay variables dropdown

    :param prawl:
    :return:
    '''

    if not prawl:

        return []

    dset = data_import.Dataset(dataset_dict[prawl])

    return dset.gen_drop_vars()

#engineering data selection
@app.callback(
    [Output('graph', 'figure'),
     Output('dtable', 'data'),
     Output('dtable', 'columns'),
     Output('t_mean', 'value')],
    [Input('select_eng', 'value'),
     Input('select_var', 'value'),
     Input('overlay_var', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')],
    State('overlay_prawler', 'value'))


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

    eng_set = data_import.Dataset(dataset_dict[dataset])
    new_data = eng_set.get_data(window_start=start_date, window_end=end_date, variables=[select_var])

    #changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if ovr_var:

        ovr_set = data_import.Dataset(dataset_dict[ovr_prawl])
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