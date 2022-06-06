'''
Gas Validaiton Dashboard

TODO:
    When the LiCOR is not calibrated it will return -50, we should filter these out by standard
    Datatable (may need to generalize the graph card for that)
    Add "select all/unselect" all button for filters
    Bug in both code and filters. See first dashboard, as it is correct.
'''

import pandas as pd

import dash
from dash import dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html as dhtml

import data_import

state_select_vars = ['INSTRUMENT_STATE', 'last_ASVCO2_validation', 'CO2LastZero', 'ASVCO2_firmware',
                     'CO2DETECTOR_serialnumber', 'ASVCO2_ATRH_serialnumber', 'ASVCO2_O2_serialnumber',
                     'ASVCO2_vendor_name']

resid_vars = ['CO2_RESIDUAL_MEAN_ASVCO2', ' CO2_DRY_RESIDUAL_MEAN_ASVCO2', ' CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2']

urls = [{'label': 'Summary Mirror', 'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/asvco2_gas_validation_summary_mirror.csv'}]

custom_sets = [{'label': 'EPOFF & APOFF vs Ref Gas',    'value': 'resids'},
               {'label': 'ZPCAL & SPPCAL vs Ref Gas',   'value': 'cals'},
               {'label': 'CO2 AVG & STDDEV',            'value': 'temp resids'},
               {'label': 'Residual vs Time',            'value': 'resid stddev'},
               {'label': 'Residual Histogram',          'value': 'stddev hist'},
               {'label': 'Summary Table',               'value': 'summary table'},
               {'label': 'Summary Data',                'value': 'summary data'}]

colors = {'Dark': {'bckgrd': '#111111', 'text': '#7FDBFF'},
          'Light': {'bckgrd': '#FAF9F6', 'text': '#111111'},
          'Green':  '#00FF00',
          'Blue':   '#0000FF',
          'Red':    '#FF0000'}

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
#                 requests_pathname_prefix='/co2/validation/',
                external_stylesheets=[dbc.themes.SLATE])
# server = app.server

filter_card = dbc.Card(
    dbc.CardBody(
        id='filter_card',
        style={'backgroundColor': colors['Dark']['bckgrd']},
        children=[dcc.DatePickerRange(id='date-picker'),
                  dcc.Checklist(id='filter1'),
                  dcc.Checklist(id='filter2'),
                  dcc.Checklist(id='filter3'),
                  dcc.Checklist(id='filter4'),
                  dcc.Checklist(id='filter5'),
                  dhtml.Button(id='update')
        ]
    )

)

tools_card = dbc.Card([
    dbc.CardBody(
           style={'backgroundColor': colors['Dark']['bckgrd']},
           children=[
            dhtml.Label(['Select Set']),
                  dcc.Dropdown(
                      id="select_set",
                      options=urls,
                      value=urls[0]['value'],
                      clearable=False
                  ),
            dhtml.Label(['Select Display']),
            dcc.Dropdown(
                id="select_display",
                options=custom_sets,
                value=custom_sets[0]['value'],
                clearable=False,
                persistence=True
                )
    ])
])

graph_card = dbc.Collapse(
    dbc.Card(
        dbc.CardBody(
            id='display-card',
            children=dcc.Graph(id='graphs'),
            style={'display': 'block'}
        )
    ),
    id='graph-collapse',
    is_open=True
)

table_card = dbc.Collapse(
    dbc.Card(
        dbc.CardBody(id='table-card',
                     children=dash_table.DataTable(id='tab1'),
                     style={'display': 'hide'}
                     )

        ),
    id='table-collapse',
    is_open=False
)


app.layout = dhtml.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([dhtml.H1('ASVCO2 Validation Set')]),
            dbc.Row([
                dbc.Col(children=[tools_card,
                                  dhtml.H5('Filters'),
                                  dcc.Loading(filter_card),
                                  dcc.RadioItems(id='image_mode',
                                                 options=['Dark', 'Light'],
                                                 value='Dark',
                                                 persistence=True)
                                  ],
                        width=3),
                dbc.Col(dcc.Loading(children=[graph_card,
                                              table_card]
                                    ),
                        width=9)
            ])
        ])
    )
])

'''
========================================================================================================================
Callbacks
'''

#changing set updates datepicker
@app.callback(
    [Output('date-picker', 'min_date_allowed'),
    Output('date-picker', 'max_date_allowed'),
    Output('date-picker', 'start_date'),
    Output('date-picker', 'end_date')],
    Input('select_set', 'value'))

def change_set(dataset_url):

    dataset = data_import.Dataset(dataset_url)

    min_date_allowed = dataset.t_start.date()
    max_date_allowed = dataset.t_end.date()
    start_date = dataset.t_start.date()
    end_date = dataset.t_end.date()

    return min_date_allowed, max_date_allowed, start_date, end_date

@app.callback(
    [Output('graph-collapse', 'is_open'),
     Output('table-collapse', 'is_open')],
     Input('select_display', 'value'))

def set_view(set_val):

    if set_val == "summary table":

        return False, True

    return True, False


# plot updating selection
@app.callback(
    [Output('display-card', 'children'),
     Output('table-card', 'children'),
     Output('filter_card', 'children')],
    [Input('select_set', 'value'),
     Input('select_display', 'value'),
     Input('image_mode', 'value'),
     Input('update', 'n_clicks')],
    [State('filter1', 'value'),
     State('filter2', 'value'),
     State('filter3', 'value'),
     State('filter4', 'value'),
     State('filter5', 'value'),
     State('date-picker', 'start_date'),
     State('date-picker', 'end_date'),
     State('tab1', 'data')
     ])

def load_plot(plot_set, plot_fig, im_mode, update, filt1, filt2, filt3, filt4, filt5, tstart, tend, table_input):

    empty_tables = dcc.Loading([dash_table.DataTable(id='tab1'), dash_table.DataTable(id='tab2')])

    def off_ref(dset, update):
        '''
        "EPOFF & APOFF vs Ref Gas"
        TODO:

        Pick serial number and last validation date
            serial: SN_ASVCO2
        Plot residuals for direct and TCOOR for EPOFF and APOFF
            state: INSTRUMENT_STATE
            residuals: CO2_RESIDUAL_MEAN_ASVCO2, CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2
            gas conc: CO2_REF_LAB
        Scatter plot
        Color based on pass/fail range check
        Hoverdata should give data summary
        :return:
        '''
        # get dataset
        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2',
                                      'OUT_OF_RANGE', 'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'SN_ASVCO2'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['EPOFF & APOFF Residual vs. Reference'],
                                   shared_yaxes=False, shared_xaxes=True)

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == []:

                return dcc.Graph(figure=load_plots), filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['OUT_OF_RANGE'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt2:
                temp.append(df[df['CO2_DRY_RESIDUAL_REF_LAB_TAG'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt3:
                temp.append(df[df['SN_ASVCO2'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

        # if we are just changing pages, then we need to refresh the filter card
        else:

            filt_list2 = []
            for var in list(df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique()):
                filt_list2.append({'label': var, 'value': var})

            filt_list3 = []
            for var in list(df['SN_ASVCO2'].unique()):
                filt_list3.append({'label': var, 'value': var})


            # default filter card
            filt_card = [dcc.DatePickerRange(id='date-picker'),
                         dhtml.Label('Out of Range'),
                         dcc.Checklist(id='filter1', options=[0, 1], value=[0, 1]),
                         dhtml.Label('Reference Range'),
                         dcc.Dropdown(id='filter2', options=filt_list2, value=df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique()[0],
                                      multi=True, clearable=True),#, persistence=True),
                         dhtml.Label('Serial #'),
                         dcc.Dropdown(id='filter3', options=filt_list3, clearable=True, multi=True,
                                      value=list(df['SN_ASVCO2'].unique())),
                         dcc.Checklist(id='filter4'),
                         dcc.Checklist(id='filter5'),
                         dhtml.Button('Update Filter', id='update')]


        epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']
        apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']

        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_RESIDUAL_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_DRY_RESIDUAL_MEAN_ASVCO2'], name='EPOFF Dry', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='EPOFF Dry TCORR', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_RESIDUAL_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_DRY_RESIDUAL_MEAN_ASVCO2'], name='APOFF Dry', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='APOFF Dry TCORR', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='Residual (ppm)',
                                    xaxis_title='Reference Gas Concentration (ppm)'
                                    )

        return dcc.Graph(figure=load_plots), empty_tables, filt_card


    def cal_ref(dset, update):
        '''
        "ZPCAL & SPPCAL vs Ref Gas"
        TODO:

        Select serial and date
            serial: SN_ASVCO2
            date: time
        Residual for ZPPCAL and SPPCAL vs gas concentration
            state: INSTRUMENT_STATE
            gas:
        :return:
        '''

        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2', 'OUT_OF_RANGE',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'SN_ASVCO2'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Cal vs Reference'],
                                   shared_yaxes=False)

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == []:

                return dcc.Graph(figure=load_plots), filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['OUT_OF_RANGE'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt2:
                temp.append(df[df['CO2_DRY_RESIDUAL_REF_LAB_TAG'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt3:
                temp.append(df[df['SN_ASVCO2'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')


        # if we are just changing pages, then we need to refresh the filter card
        else:

            filt_list2 = []
            for var in list(df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique()):
                filt_list2.append({'label': var, 'value': var})

            filt_list3 = []
            for var in list(df['SN_ASVCO2'].unique()):
                filt_list3.append({'label': var, 'value': var})

            filt_card = [dcc.DatePickerRange(id='date-picker'),
                         dhtml.Label('Out of Range'),
                         dcc.Checklist(id='filter1', options=[0, 1], value=[0, 1]),
                         dhtml.Label('Reference Range'),
                         dcc.Dropdown(id='filter2', options=filt_list2,
                                      value=df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique()[0],
                                      multi=True, clearable=True),# persistence=True),
                         dhtml.Label('Serial #'),
                         dcc.Dropdown(id='filter3', options=filt_list3, clearable=True, multi=True,
                                      value=list(df['SN_ASVCO2'].unique())),
                         dcc.Checklist(id='filter4'),
                         dcc.Checklist(id='filter5'),
                         dhtml.Button('Update Filter', id='update')]


        zcal = df[df['INSTRUMENT_STATE'] == 'ZPPCAL']
        scal = df[df['INSTRUMENT_STATE'] == 'SPPCAL']

        load_plots.add_scatter(x=zcal['CO2_REF_LAB'], y=zcal['CO2_RESIDUAL_MEAN_ASVCO2'], name='ZPPCAL',
                               hoverinfo='x+y+name', mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=scal['CO2_REF_LAB'], y=scal['CO2_RESIDUAL_MEAN_ASVCO2'], name='SPPCAL',
                               hoverinfo='x+y+name', mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=zcal['CO2_REF_LAB'], y=zcal['CO2_DRY_RESIDUAL_MEAN_ASVCO2'], name='ZPPCAL Dry',
                               hoverinfo='x+y+name', mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=scal['CO2_REF_LAB'], y=scal['CO2_DRY_RESIDUAL_MEAN_ASVCO2'], name='SPPCAL Dry',
                               hoverinfo='x+y+name', mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=zcal['CO2_REF_LAB'], y=zcal['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='ZPPCAL TCORR',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=scal['CO2_REF_LAB'], y=scal['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='SPPCAL TCORR',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(
                                    yaxis_title='Residual (ppm)',
                                    xaxis_title='Reference Gas (ppm)',
                                    )

        return dcc.Graph(figure=load_plots), empty_tables, filt_card

    def multi_ref(dset, update):
        '''
        "CO2 AVG & STDDEV"
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Residual
        :return:
        TODO:

        '''


        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_RESIDUAL_STDDEV_ASVCO2', 'SN_ASVCO2', 'ASVCO2_firmware',
                                      'CO2DETECTOR_firmware', 'last_ASVCO2_validation'])

        load_plots = make_subplots(rows=2, cols=1,
                                   subplot_titles=['Multi-unit Residual vs Ref Gas'],
                                   shared_yaxes=False)

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == [] or filt4 == []:
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return load_plots, filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['SN_ASVCO2'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt2:
                temp.append(df[df['CO2DETECTOR_firmware'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt3:
                temp.append(df[df['ASVCO2_firmware'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt4:
                temp.append(df[df['last_ASVCO2_validation'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

        else:

            filt_list1 = []
            filt_list3 = []
            filt_list4 = []

            for var in df['SN_ASVCO2'].unique():
                filt_list1.append({'label': var, 'value': var})
            for var in df['ASVCO2_firmware'].unique():
                filt_list3.append({'label': var, 'value': var})
            for var in list(df['last_ASVCO2_validation'].unique()):
                filt_list4.append({'label': var, 'value': var})

            filt_card = [dcc.DatePickerRange(id='date-picker'),
                         dhtml.Label('Serial #'),
                         dcc.Dropdown(id='filter1', options=filt_list1, clearable=True, multi=True,
                                     value=list(df['SN_ASVCO2'].unique())),
                         dhtml.Label('LiCOR Firmware'),
                         dcc.Checklist(id='filter2', options=list(df['CO2DETECTOR_firmware'].unique()),
                                     value=list(df['CO2DETECTOR_firmware'].unique())),
                         dhtml.Label('ASVCO2 firmware'),
                         dcc.Dropdown(id='filter3', options=filt_list3, multi=True, clearable=True,
                                       value=list(df['ASVCO2_firmware'].unique())),
                         dhtml.Label('Last Validation'),
                         dcc.Dropdown(id='filter4', options=filt_list4, value=list(df['last_ASVCO2_validation'].unique()),
                                      multi=True, clearable=True),#persistence=True,
                         dcc.Checklist(id='filter5'),
                         dhtml.Button('Update Filter', id='update')]

        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_RESIDUAL_MEAN_ASVCO2'], name='Residual', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_DRY_RESIDUAL_MEAN_ASVCO2'], name='Dry Resid', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='TCORR Dry Resid', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_RESIDUAL_STDDEV_ASVCO2'], name='Residual STDDEV',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=2, col=1)

        load_plots['layout'].update(yaxis1_title='CO2 Residuals (ppm)',
                                    xaxis1_title='CO2 Gas Concentration (ppm)',
                                    yaxis2_title='CO2 STDDEV (ppm)',
                                    xaxis2_title='CO2 Gas Concentration (ppm)',
                                    )


        return dcc.Graph(figure=load_plots), empty_tables, filt_card


    def resid_and_stdev(dset, update):
        '''
        "Resid vs Time"
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Residual vs time, with STDDEV as error bars
        :return:

        TODO:
            Filter by CO2_DRY_RESIDUAL_REF_LAB_TAG

        NOTES:
            At test, the Last Validation filter doesn't appear to work.
        '''

        df = dset.get_data(variables=['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'CO2_RESIDUAL_STDDEV_ASVCO2',
                                      'SN_ASVCO2', 'ASVCO2_firmware', 'CO2DETECTOR_firmware', 'last_ASVCO2_validation'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Residual Over Time'],
                                   shared_yaxes=False)

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == [] or filt4 == [] or filt5:
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return load_plots, filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['SN_ASVCO2'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt2:
                temp.append(df[df['CO2DETECTOR_firmware'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt3:
                temp.append(df[df['ASVCO2_firmware'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt4:
                temp.append(df[df['last_ASVCO2_validation'] == var])

            temp = []

            for var in filt5:
                temp.append(df[df['INSTRUMENT_STATE'] == var])

            df = pd.merge(df, pd.concat(temp), how='right')


        else:

            filt_list1 = []
            filt_list3 = []
            filt_list4 = []

            for var in df['SN_ASVCO2'].unique():
                filt_list1.append({'label': var, 'value': var})
            for var in df['ASVCO2_firmware'].unique():
                filt_list3.append({'label': var, 'value': var})
            for var in df['last_ASVCO2_validation'].unique():
                filt_list4.append({'label': var, 'value': var})

            filt_card = [dcc.DatePickerRange(id='date-picker'),
                         dhtml.Label('Serial #'),
                         dcc.Dropdown(id='filter1', options=filt_list1, multi=True, clearable=True,
                                       value=list(df['SN_ASVCO2'].unique())),
                         dhtml.Label('LiCOR Firmware'),
                         dcc.Checklist(id='filter2', options=df['CO2DETECTOR_firmware'].unique(),
                                       value=list(df['CO2DETECTOR_firmware'].unique())),
                         dhtml.Label('ASVCO2 firmware'),
                         dcc.Dropdown(id='filter3', options=filt_list3, multi=True, clearable=True,
                                       value=list(df['ASVCO2_firmware'].unique())),
                         dhtml.Label('Last Validation'),
                         dcc.Dropdown(id='filter4', options=filt_list4, multi=True, clearable=True,
                                       value=list(df['last_ASVCO2_validation'].unique())),
                         dhtml.Label('Instrument State'),
                         dcc.Checklist(id='filter5', options=['APOFF', 'EPOFF'],
                                       value=['APOFF', 'EPOFF']),
                         dhtml.Button('Update Filter', id='update')]

        # plotting block

        load_plots.add_scatter(x=df['time'], y=df['CO2_RESIDUAL_MEAN_ASVCO2'], error_y=dict(array=df['CO2_RESIDUAL_STDDEV_ASVCO2']),
                               name='Residual', hoverinfo='x+y+name', mode='markers', marker={'size': 4}, row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['CO2_DRY_RESIDUAL_MEAN_ASVCO2'], error_y=dict(array=df['CO2_RESIDUAL_STDDEV_ASVCO2']),
                               name='Dry Residual', hoverinfo='x+y+name', mode='markers', marker={'size': 4}, row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], error_y=dict(array=df['CO2_RESIDUAL_STDDEV_ASVCO2']),
                               name='Dry TCORR Residual', hoverinfo='x+y+name', mode='markers', marker={'size': 4}, row=1, col=1)

        load_plots['layout'].update(
            yaxis_title='Residual w/ Standard Deviation (ppm)'
        )

        return dcc.Graph(figure=load_plots), empty_tables, filt_card

    def stddev_hist(dset, update):
        '''
        "Residual Histogram"
        Select random variable
        Histogram of marginal probability dists

        TODO:

        :return:
        '''

        resid_sets = ['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2']

        df = dset.get_data(variables=['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'ASVCO2_firmware',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'INSTRUMENT_STATE', 'CO2DETECTOR_firmware',
                                      'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'last_ASVCO2_validation'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Residual Histogram'],
                                   shared_yaxes=False)

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == [] or filt4 == []:
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return load_plots, filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['ASVCO2_firmware'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt2:
                temp.append(df[df['INSTRUMENT_STATE'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt3:
                temp.append(df[df['CO2_DRY_RESIDUAL_REF_LAB_TAG'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            temp = []

            for var in filt4:
                temp.append(df[df['last_ASVCO2_validation'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

            for co2_set in filt1:
                load_plots.add_trace(go.Histogram(x=df[co2_set], name=co2_set, xbins=dict(size=1)), row=1, col=1)

        else:

            filt_list1 = []
            filt_list3 = []
            filt_list4 = []

            for var in df['ASVCO2_firmware'].unique():
                filt_list1.append({'label': var, 'value': var})
            for var in df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique():
                filt_list3.append({'label': var, 'value': var})
            for var in df['last_ASVCO2_validation'].unique():
                filt_list4.append({'label': var, 'value': var})

            filt_card = [dcc.DatePickerRange(id='date-picker'),
                         dhtml.Label('ASVCO2 firmware'),
                         dcc.Dropdown(id='filter1', options=filt_list1, multi=True, clearable=True,
                                      value=list(df['ASVCO2_firmware'].unique())),
                         dhtml.Label('Instrument State'),
                         dcc.Checklist(id='filter2', options=['APOFF', 'EPOFF'],
                                       value=['APOFF', 'EPOFF']),
                         dhtml.Label('Residual Reference Tag'),
                         dcc.Dropdown(id='filter3', options=filt_list3, multi=True, clearable=True,
                                       value=df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique()),
                         dhtml.Label('Last Validation'),
                         dcc.Dropdown(id='filter4', options=filt_list4, multi=True, clearable=True,
                                      value=list(df['last_ASVCO2_validation'].unique())),
                         dhtml.Label('LiCOR Firmware'),
                         dcc.Checklist(id='filter5', options=df['CO2DETECTOR_firmware'].unique(),
                                       value=list(df['CO2DETECTOR_firmware'].unique())),
                         dhtml.Button('Update Filter', id='update')]

            for co2_set in resid_sets:
                load_plots.add_trace(go.Histogram(x=df[co2_set], name=co2_set, xbins=dict(size=1)), row=1, col=1)

        load_plots['layout'].update(
            yaxis_title='n readings',
            xaxis_title='Residuals (ppm)',
            xaxis=dict(range=[-50, 50])
        )

        return dcc.Graph(figure=load_plots), empty_tables, filt_card


    def summary_table(dset, update):
        '''
        Returns

        :return:
        '''


        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2', 'SN_ASVCO2',
                                 'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'ASVCO2_firmware',
                                 'CO2_RESIDUAL_STDDEV_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'OUT_OF_RANGE'])

        df = df[(df['INSTRUMENT_STATE'] == 'APOFF') | (df['INSTRUMENT_STATE'] == 'EPOFF')]

        limiters = {'range_min':     0,
                    'range_max':     2,
                    'pf_mean':       1,
                    'pf_stddev':     .5,
                    'pf_max':        2
                    }

        default = [{'sn': 'Min', 'mean': 0, 'stddev': '', 'max': ''},  # defaults
                   {'sn': 'Max', 'mean': 2, 'stddev': '', 'max': ''},
                   {'sn': '', 'mean': 'Mean Residual', 'stddev': 'STDDEV', 'max': 'Max Residual'},
                   {'sn': 'Pass/Fail', 'mean': 1, 'stddev': .5, 'max': 2},  # defaults
                   {'sn': 'APOFF Fail %', 'mean': '', 'stddev': '', 'max': ''},
                   {'sn': 'EPOFF Fail %', 'mean': '', 'stddev': '', 'max': ''}
                   ]


        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # supposedly dash_tables will have built-in typing at some point in the future... this might make them even
            # more of a mess on the backend, but will help eliminate this mess of a guard clause

            if table_input[0]['mean'] != '':
                try:
                    limiters['range_min'] = float(table_input[0]['mean'])
                    default[0]['mean'] = float(table_input[0]['mean'])
                except (ValueError, TypeError):
                    default[0]['mean'] = ''
                    limiters['range_min'] = None

            if table_input[1]['mean'] != '':
                try:
                    limiters['range_max'] = float(table_input[1]['mean'])
                    default[1]['mean'] = float(table_input[1]['mean'])
                except (ValueError, TypeError):
                    limiters['range_max'] = None
                    default[1]['mean'] = ''

            if table_input[3]['mean'] != '':
                try:
                    limiters['pf_mean'] = float(table_input[3]['mean'])
                    default[3]['mean'] = float(table_input[3]['mean'])
                except (ValueError, TypeError):
                    limiters['pf_mean'] = None
                    default[3]['mean'] = ''

            if table_input[3]['stddev'] != '':
                try:
                    limiters['pf_stddev'] = float(table_input[3]['stddev'])
                    default[3]['stddev'] = float(table_input[3]['stddev'])
                except (ValueError, TypeError):
                    limiters['pf_stddev'] = None
                    default[3]['stddev'] = ''

            if table_input[3]['max'] != '':
                try:
                    limiters['pf_max'] = float(table_input[3]['max'])
                    default[3]['max'] = float(table_input[3]['max'])
                except (ValueError, TypeError):
                    limiters['pf_max'] = None
                    default[3]['max'] = ''

            # filter for Instrument state
            temp = []

            for var in ['APOFF', 'EPOFF']:
                temp.append(df[df['INSTRUMENT_STATE'] == var])
            df = pd.merge(df, pd.concat(temp), how='right')

        else:

            filt_list1 = [{'label': 'Dry Residual',    'value': 'CO2_DRY_RESIDUAL_MEAN_ASVCO2'},
                          {'label': 'TCORR Residual',  'value': 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'},
                          {'label': 'STDDEV', 'value': 'CO2_RESIDUAL_STDDEV_ASVCO2'}
                          ]

            #filt_list2 = []
            filt_list3 = []

            # for state in df['INSTRUMENT_STATE'].unique():
            #     filt_list2.append({'label': state, 'value': state})

            for rng in df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique():
                filt_list3.append({'label': rng, 'value': rng})

            filt_card = [dcc.DatePickerRange(id='date-picker'),
                         dhtml.Label('Type'),
                         dcc.Dropdown(id='filter1', options=filt_list1, value='CO2_DRY_RESIDUAL_MEAN_ASVCO2',
                                      clearable=False, multi=False),
                         dhtml.Label('State'),
                         #dcc.Dropdown(id='filter2', options=filt_list2, value='APOFF', clearable=False, multi=False),
                         dcc.Checklist(id='filter2', options=['APOFF', 'EPOFF'], value=['APOFF', 'EPOFF']),
                         dhtml.Label('Residual Lab Reference Range'),
                         dcc.Dropdown(id='filter3', options=filt_list3, clearable=True, value=None),
                         dcc.Checklist(id='filter4'),
                         dcc.Checklist(id='filter5'),
                         dhtml.Button('Update Filter', id='update')]

        table_data = dict()
        temp = {'Both': dict(),
                'APOFF': dict(),
                'EPOFF': dict()}

        # default values
        if filt1 == []:
            filt1 = 'CO2_DRY_RESIDUAL_MEAN_ASVCO2'

        if (filt2 == []) or (filt2 == ['APOFF', 'EPOFF']):
            filt2 = 'Both'

        if limiters['range_min']:
            df = df[df['CO2_REF_LAB'] > limiters['range_min']]

        if limiters['range_max']:
            df = df[df['CO2_REF_LAB'] < limiters['range_max']]

        for sn in df['SN_ASVCO2'].unique():

            if sn in df['SN_ASVCO2'].unique():
                temp['Both'][sn] = {'sn': sn,
                            'mean': df[df['SN_ASVCO2'] == sn][filt1].mean(),
                            'stddev': df[df['SN_ASVCO2'] == sn][filt1].std(),
                            'max': df[df['SN_ASVCO2'] == sn][filt1].max()}

            else:
                temp['Both'][sn] = {'sn': sn,
                            'mean': '',
                            'stddev': '',
                            'max': ''}

            for state in ['APOFF', 'EPOFF']:

                if sn in df['SN_ASVCO2'].unique():
                    temp[state][sn] = {'sn': sn,
                                        'mean': df[(df['SN_ASVCO2'] == sn) &
                                                   (df['INSTRUMENT_STATE'] == state)][filt1].mean(),
                                        'stddev': df[(df['SN_ASVCO2'] == sn) &
                                                     (df['INSTRUMENT_STATE'] == state)][filt1].std(),
                                        'max': df[(df['SN_ASVCO2'] == sn) &
                                                  (df['INSTRUMENT_STATE'] == state)][filt1].max()}

                else:
                    temp[state][sn] = {'sn': sn,
                                        'mean': '',
                                        'stddev': '',
                                        'max': ''}

        table_data = {'Both':   pd.DataFrame.from_dict(temp['Both'], orient='index'),
                      'APOFF':  pd.DataFrame.from_dict(temp['APOFF'], orient='index'),
                      'EPOFF':  pd.DataFrame.from_dict(temp['EPOFF'], orient='index')}

        # Mean pass/fail percentage
        # both = table_data['Both'].dropna(subset=filt1)
        # apoff = table_data['APOFF'].dropna(subset=filt1)
        # epoff = table_data['EPOFF'].drnopna(subset=filt1)


        if table_data[filt2].dropna(subset='mean').empty:
            default[4]['mean'], default[5]['mean'] = '', ''

        elif limiters['pf_mean']:

            temp = table_data['APOFF'].dropna(subset='mean')

            #perc = 100 * len(table_data["APOFF"][abs(table_data["APOFF"]['mean']) > limiters["pf_mean"]]) / len(table_data["APOFF"])
            perc = 100 * len(temp[abs(temp['mean']) > limiters["pf_mean"]]) / len(temp)

            default[4]['mean'] = f'{perc}%'

            temp = table_data['EPOFF'].dropna(subset='mean')

            #perc = 100 * len(abs(table_data["EPOFF"][table_data["EPOFF"]['mean'] > limiters["pf_mean"]])) / len(table_data["EPOFF"])
            perc = 100 * len(temp[abs(temp['mean']) > limiters["pf_mean"]]) / len(temp)

            default[5]['mean'] = f'{perc}%'

        # STDDEV pass/fail percentage
        if table_data[filt2].dropna(subset='stddev').empty:
            default[4]['stddev'], default[5]['stddev'] = '', ''

        elif limiters['pf_stddev']:

            temp = table_data['APOFF'].dropna(subset='stddev')

            #perc = 100 * len(abs(table_data["APOFF"][table_data["APOFF"]['stddev'] > limiters["pf_stddev"]])) / len(table_data["APOFF"])
            perc = 100 * len(temp[abs(temp['stddev']) > limiters["pf_stddev"]]) / len(temp)

            default[4]['stddev'] = f'{perc}%'

            temp = table_data['EPOFF'].dropna(subset='stddev')

            #perc = 100 * len(abs(table_data["EPOFF"][table_data["EPOFF"]['stddev'] > limiters["pf_stddev"]])) / len(table_data["EPOFF"])
            perc = 100 * len(temp[abs(temp['stddev']) > limiters["pf_stddev"]]) / len(temp)

            default[5]['stddev'] = f'{perc}%'

        # Max pass/fail percentage
        if table_data[filt2].dropna(subset='max').empty:
            default[4]['max'], default[5]['max'] = '', ''

        elif limiters['pf_max']:

            temp = table_data['APOFF'].dropna(subset='max')

            #perc = 100 * len(abs(table_data["APOFF"][table_data["APOFF"]['max'] > limiters["max"]])) / len(table_data["APOFF"])
            perc = 100 * len(temp[abs(temp['max']) > limiters["pf_max"]]) / len(temp)

            default[4]['max'] = f'{perc}%'

            temp = table_data['EPOFF'].dropna(subset='max')

            #perc = 100 * len(abs(table_data["EPOFF"][table_data["EPOFF"]['max'] > limiters["max"]])) / len(table_data["EPOFF"])
            perc = 100 * len(temp[abs(temp['max']) > limiters["pf_max"]]) / len(temp)

            default[5]['max'] = f'{perc}%'

        def_df = pd.DataFrame.from_dict(default)
        def_df.fillna('')

        tab1 = dash_table.DataTable(def_df.to_dict('records'),
                                    columns=[{"name": i, "id": i, 'editable': True} for i in def_df.columns],
                                    id='tab1',
                                    style_table={'backgroundColor': colors[im_mode]['bckgrd']},
                                    style_cell={'backgroundColor': colors[im_mode]['bckgrd'],
                                                'textColor': colors[im_mode]['text']}
                                    )

        tab2 = dash_table.DataTable(table_data[filt2].to_dict('records'),
                                    columns=[{"name": i, "id": i} for i in table_data[filt2].columns],
                                    id='tab2',
                                    style_table={'backgroundColor': colors[im_mode]['bckgrd']},
                                    style_cell={'backgroundColor': colors[im_mode]['bckgrd'],
                                                'textColor': colors[im_mode]['text']}
                                    )

        return dcc.Graph(id='graphs'), [dcc.Loading(tab1), dcc.Loading(tab2)], filt_card

    def summary(dset, update):
        '''
        Returns

        :return:
        '''

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Residual'],
                                   shared_yaxes=False)

        df = dset.get_data(variables=['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_RESIDUAL_STDDEV_ASVCO2', 'CO2_REF_LAB',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'INSTRUMENT_STATE'])


        return dcc.Graph(figure=load_plots), dash.no_update

    # enforce filter return as list
    if not isinstance(filt1, list):
        filt1 = [filt1]
    if not isinstance(filt2, list):
        filt2 = [filt2]
    if not isinstance(filt3, list):
        filt3 = [filt3]
    if not isinstance(filt4, list):
        filt4 = [filt4]

    def switch_plot(case):
        return {'resids':        off_ref,
                'cals':          cal_ref,
                'temp resids':   multi_ref,
                'resid stddev':  resid_and_stdev,
                'stddev hist':   stddev_hist,
                'summary table': summary_table,
                'summary data': summary
                }.get(case)

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

    dataset = data_import.Dataset(plot_set, window_start=tstart, window_end=tend)
    plotters = switch_plot(plot_fig)(dataset, update)

    if plot_fig == 'summary table':
        pass

    else:
        plotters[0].figure.update_layout(height=600,
            title=' ',
            #hovermode='x unified',
            xaxis_showticklabels=True,
            yaxis_fixedrange=False,
            plot_bgcolor=colors[im_mode]['bckgrd'],
            paper_bgcolor=colors[im_mode]['bckgrd'],
            font_color=colors[im_mode]['text'],
            yaxis_gridcolor=colors[im_mode]['text'],
            xaxis_gridcolor=colors[im_mode]['text'],
            yaxis_zerolinecolor=colors[im_mode]['text'],
            xaxis_zerolinecolor=colors[im_mode]['text'],
            #autosize=True,
            #xaxis=dict(showgrid=False),
            showlegend=True, modebar={'orientation': 'h'},
            margin=dict(l=25, r=25, b=25, t=25, pad=4)
        )

    return plotters[0], plotters[1], plotters[2]


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)
