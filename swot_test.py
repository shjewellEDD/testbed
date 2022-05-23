'''
TODO:
    Improvement:
        Things could be simplified by adding overlay data as a column to the plot data, instead of using seperate
        DataFrame
        Color:
            Add color to SB depth
            Let user select color driver from dropdown
        Date errors:
            M200 science has a bunch of dates from the 70s and 80s
            How do we deal with this
            Just drop?
            Linearly interpolate?
    Bugs:


'''

import dash
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

prawlers = [{'label':   'M200 Eng', 'value': 'M200Eng'},
            {'label':   'M200 Sci', 'value': 'M200Sci'}
            ]

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
                requests_pathname_prefix='/swot/test/',
                external_stylesheets=[dbc.themes.SLATE])
server = app.server

external_stylesheets = ['https://codepen.io./chriddyp/pen/bWLwgP.css']

set_card = dbc.Card([
        dbc.CardBody(
            children=[
                dhtml.H5('Plot'),
                dcc.Dropdown(
                    id="select_eng",
                    options=prawlers,
                    value=prawlers[0]['value'],
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
                    options=prawlers,
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
        dcc.DatePickerRange(
            id='date-picker',
            style={'backgroundColor': colors['background']},
        ),
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
                                                    'height'         :'300px',
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
                dbc.Col(graph_card, width=9),
                dbc.Col(children=[date_card,
                                  set_card,
                                  overlay_card,
                                  table_card],
                        width=3)
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

    eng_set = data_import.Dataset(dataset_dict[dataset])

    min_date_allowed = eng_set.t_start.date()
    max_date_allowed = eng_set.t_end.date()
    start_date = (eng_set.t_end - datetime.timedelta(days=14)).date()
    end_date = eng_set.t_end.date()
    first_var = eng_set.ret_vars()[0]

    return eng_set.ret_vars(), min_date_allowed, max_date_allowed, start_date, end_date, first_var


#overlay selection
@app.callback(
    Output('overlay_var', 'options'),
    Input('overlay_prawler', 'value')
)

def overlay_vars(prawl):

    if not prawl:

        return []

    dset = data_import.Dataset(dataset_dict[prawl])

    return dset.ret_vars()

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

    #elif select_var in list(new_data.columns):

    else:
        if ovr_var:

            efig = px.scatter(y=new_data[select_var], x=new_data['time'], color=ovr_data[ovr_var],
                              color_continuous_scale=colorscale)
            efig.layout['coloraxis']['colorbar']['title']['text'] = ovr_var

        else:
            efig = px.scatter(new_data, y=select_var, x='time')

        columns = [{"name": 'Date', "id": 'datetime'},
                   {'name': select_var, 'id': select_var}]

        try:
            t_mean = 'Average ' + select_var + ': ' + str(round(new_data.loc[:, select_var].mean(), 3))
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
        font_color=colors['text']
    )

    return efig, table_data, columns, t_mean


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)
    app.run_server(debug=True)