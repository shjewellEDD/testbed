'''
TODO:
    Improvement:
        Color:
            Probably should update_layout wiht color instead of px.scatter
        Data selection:
            Break up Prawler from set
        Profile plotting, see Scott's email
        Improve style sheet
        Generalize plotting, so I don't need separate functions to do it.
        Date errors:
            M200 science has a bunch of dates from the 70s and 80s
            How do we deal with this
            Just drop?
            Linearly interpolate?
    Bugs:
        TELONAS2 GEN data ends in 4/21, but dateselector doesn't update to reflect this

'''

import dash
from dash import html as dhtml
from dash import dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
# import plotly.graph_objects as go
import dash_bootstrap_components as dbc

#non-plotly imports
import pandas as pd
import datetime
import requests


prawlers = [{'label':   'Eng', 'value': 'Eng'},
            {'label':   'Sci', 'value': 'Sci'},
            {'label':   'Load', 'value': 'Load'},
            {'label':   'Baro', 'value': 'Baro'}
            ]


skipvars = ['time', 'Time', 'TIME', 'latitude', 'longitude', 'timeseries_id', 'profile_id', 'Epoch_Time']


# will need to be altered for multi-set displays

# ======================================================================================================================
# helpful functions

# generates ERDDAP compatable date
def gen_erddap_date(edate):
    erdate = (str(edate.year) + "-"
              + str(edate.month).zfill(2) + '-'
              + str(edate.day).zfill(2) + "T"
              + str(edate.hour).zfill(2) + ":"
              + str(edate.minute).zfill(2) + ":"
              + str(edate.second).zfill(2) + "Z")

    return erdate


# generates datetime.datetime object from ERDDAP compatable date
def from_erddap_date(edate):
    redate = datetime.datetime(year=int(edate[:4]),
                               month=int(edate[5:7]),
                               day=int(edate[8:10]),
                               hour=int(edate[11:13]),
                               minute=int(edate[14:16]),
                               second=int(edate[17:19]))

    return redate

''' class used for holding useful information about the ERDDAP databases
========================================================================================================================
'''

class Dataset:
    #dataset object,
    #it takes requested data and generates windows and corresponding urls
    #logger.info('New dataset initializing')

    def __init__(self, url, window_start=False, window_end=False):
        self.url = url
        self.t_start, self.t_end = self.data_dates()
        self.data, self.vars = self.get_data()

    #opens metadata page and returns start and end datestamps
    def data_dates(self):
        page = (requests.get(self.url[:-3] + "das")).text

        indx = page.find('Float64 actual_range')
        mdx = page.find(',', indx)
        endx = page.find(";", mdx)
        start_time = datetime.datetime.utcfromtimestamp(float(page[(indx + 21):mdx]))
        end_time = datetime.datetime.utcfromtimestamp(float(page[(mdx + 2):endx]))

        #prevents dashboard from trying to read data from THE FUTURE!
        if end_time > datetime.datetime.now():
            end_time = datetime.datetime.now()

        return start_time, end_time


    def get_data(self):

        self.data = pd.read_csv(self.url, skiprows=[1])
        dat_vars = self.data.columns

        # for set in list(data.keys()):
        self.vars = []
        for var in list(dat_vars):
            if var in skipvars:
                continue

            self.vars.append({'label': var, 'value': var.lower()})

        vars_lower = [each_str.lower() for each_str in dat_vars]

        if 'nerrors' in vars_lower:
            self.vars.append({'label': 'Errors Per Day', 'value': 'errs_per_day'})
        if 'ntrips' in vars_lower:
            self.vars.append({'label': 'Trips Per Day', 'value': 'trips_per_day'})
        if 'sb_depth' in vars_lower:
            self.vars.append({'label': 'Sci Profiles Per Day', 'value': 'sci_profs'})

        self.data.columns = self.data.columns.str.lower()

        if 'dir' in list(self.data.columns):
            if not self.data[self.data['dir'] == 'F'].empty:

                self.vars.append({'label': 'Failures', 'value': 'failures'})
                self.vars.append({'lable': 'Time to Failure', 'value': 'time_to_fail'})

        self.data['datetime'] = self.data['time'].apply(from_erddap_date)
        self.data.drop(self.data[self.data['datetime'] > datetime.datetime.today()].index, axis='rows')

        return self.data, self.vars

    def ret_data(self, w_start, w_end):

        #self.data['datetime'] = self.data.loc[:, 'time'].apply(from_erddap_date)

        return self.data[(w_start <= self.data['datetime']) & (self.data['datetime'] <= w_end)]

    def ret_vars(self):

        return self.vars

    def trips_per_day(self, w_start, w_end):

        internal_set =self.data[(w_start <= self.data['datetime']) & (self.data['datetime'] <= w_end)]
        #internal_set['datetime'] = internal_set.loc[:, 'time'].apply(from_erddap_date)
        internal_set['days'] = internal_set.loc[:, 'datetime'].dt.date
        new_df = pd.DataFrame((internal_set.groupby('days')['ntrips'].last()).diff())[1:]
        new_df['days'] = new_df.index

        return new_df

    def errs_per_day(self, w_start, w_end):

        internal_set = self.data[(w_start <= self.data['datetime']) & (self.data['datetime'] <= w_end)]
        # internal_set['datetime'] = internal_set.loc[:, 'time'].apply(from_erddap_date)
        internal_set['days'] = internal_set.loc[:, 'datetime'].dt.date
        new_df = pd.DataFrame((internal_set.groupby('days')['nerrors'].last()).diff())[1:]
        new_df['days'] = new_df.index

        return new_df

    def gen_fail_set(self):

        fail_set = self.data[self.data['dir'] == 'F']
        # fail_set['datetime'] = fail_set.loc[:, 'time'].apply(from_erddap_date)
        fail_set['days'] = fail_set.loc[:, 'datetime'].dt.date
        fail_set = pd.DataFrame((fail_set.groupby('days')['dir'].last()).diff())[1:]
        fail_set['days'] = fail_set.index

        return fail_set

    def sci_profiles_per_day(self, w_start, w_end):

        sci_set = self.data[self.data.loc[:, 'sb_depth'].diff() < -35]
        sci_set['ntrips'] = sci_set['sb_depth'].diff()

        # sci_set['datetime'] = sci_set.loc[:, 'time'].apply(from_erddap_date)
        sci_set['days'] = sci_set.loc[:, 'datetime'].dt.date
        sci_set = pd.DataFrame((sci_set.groupby('days')['ntrips'].size()))
        sci_set['days'] = sci_set.index

        return sci_set
'''
========================================================================================================================
Start Dashboard
'''

dataset_dict = {
            'Eng': Dataset('https://data.pmel.noaa.gov/engineering/erddap/tabledap/prawler_eng_TELONAS2.csv'),
            'Sci': Dataset('https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELONAS2_PRAWC_NAS2.csv'),
            'Load': Dataset('https://data.pmel.noaa.gov/engineering/erddap/tabledap/TELONAS2_LOAD_NAS2.csv'),
            'Baro': Dataset('https://data.pmel.noaa.gov/engineering/erddap/tabledap/prawler_baro_TELONAS2.csv')
            }


#eng_set = Dataset(set_meta['Eng']['url'])
starting_set = 'Eng'

graph_config = {'modeBarButtonsToRemove' : ['hoverCompareCartesian','select2d', 'lasso2d'],
                'doubleClick':  'reset+autosize', 'toImageButtonOptions': { 'height': None, 'width': None, },
                'displaylogo': False}

colors = {'background': '#111111', 'text': '#7FDBFF'}

external_stylesheets = ['https://codepen.io./chriddyp/pen/bWLwgP.css']

variables_card = dbc.Card(
    [#dbc.CardHeader("Tools"),
     dbc.CardBody(
         dcc.Dropdown(
             id="select_var",
             #style={'backgroundColor': colors['background']},
             #       'textColor': colors['text']},
             options=dataset_dict[starting_set].ret_vars(),
             value=dataset_dict[starting_set].ret_vars()[0]['value'],
             clearable=False
         ),
    )],
    color='dark'
)

set_card = dbc.Card([
        dbc.CardBody(
            dcc.Dropdown(
                id="select_eng",
                #style={'backgroundColor': colors['background']},
                options=prawlers,
                value=prawlers[0]['value'],
                clearable=False
            )
        )
])

date_card = dbc.Card([
    dbc.CardBody(
        dcc.DatePickerRange(
            id='date-picker',
            style={'backgroundColor': colors['background']},
            min_date_allowed=dataset_dict[starting_set].t_start.date(),
            max_date_allowed=dataset_dict[starting_set].t_end.date(),
            start_date=(dataset_dict[starting_set].t_end - datetime.timedelta(days=14)).date(),
            end_date=dataset_dict[starting_set].t_end.date()
        ),
    )
])

table_card = dbc.Card([
    dbc.CardBody(
        children=[dcc.Textarea(id='t_mean',
                                value='',
                                readOnly=True,
                                style={'width': '100%', 'height': 40,
                                        #'backgroundColor': colors['background'],
                                        'textColor':       colors['text']},
                                ),
                    dash_table.DataTable(id='table',
                                         style_table={'backgroundColor': colors['background'],
                                                      'height'         :'300px',
                                                      'overflowY'       :'auto'},
                                                      #'overflow'      : 'scroll'},
                                         style_cell={'backgroundColor': colors['background'],
                                                     'textColor':       colors['text']}
                    )
        ])
])

graph_card = dbc.Card(
    [#dbc.CardHeader("Here's a graph"),
     dbc.CardBody([dcc.Graph(id='graph')
                   ])
    ]
)


app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                requests_pathname_prefix='/prawler/telonas2/',
                external_stylesheets=[dbc.themes.SLATE])
#server = app.server

app.layout = dhtml.Div([
   #dbc.Container([
            dbc.Row([dhtml.H1('Prawler TELONAS2')]),
            dbc.Row([
                dbc.Col(graph_card, width=9),
                dbc.Col(children=[date_card,
                                  set_card,
                                  variables_card,
                                  table_card],
                        width=3)
                    ])
   #             ])
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

    eng_set = dataset_dict[dataset]

    min_date_allowed = eng_set.t_start.date(),
    max_date_allowed = eng_set.t_end.date(),
    start_date = (eng_set.t_end - datetime.timedelta(days=14)).date(),
    end_date = eng_set.t_end.date()
    first_var = eng_set.ret_vars()[0]['value']

    return dataset_dict[dataset].ret_vars(), str(min_date_allowed[0]), str(max_date_allowed[0]), str(start_date[0]), str(end_date), first_var


#engineering data selection
@app.callback(
    [Output('graph', 'figure'),
     Output('table', 'data'),
     Output('table', 'columns'),
     Output('t_mean', 'value')],
    [Input('select_eng', 'value'),
     Input('select_var', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])

def plot_evar(dataset, select_var, start_date, end_date):

    eng_set = dataset_dict[dataset]
    new_data = eng_set.ret_data(start_date, end_date)

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
        efig = px.scatter(new_data, y=select_var, x='time')#, color="sepal_length", color_continuous_scale=colorscale)

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
        font_color=colors['text'],
    )

    return efig, table_data, columns, t_mean


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)
    app.run_server(debug=True)