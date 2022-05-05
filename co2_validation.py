'''
Gas Validaiton Dashboard

TODO:
    Check all titles and axes labels
    See if can color by subset
    Get histogram names working
    Datatable (may need to generalize the graph card for that)
    Add "select all/unselect" all button for filters
    Add "Refresh" button
'''

import datetime
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

resid_vars = ['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_RESIDUAL_STDDEV_ASVCO2', ' CO2_DRY_RESIDUAL_MEAN_ASVCO2', ' CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2',
              ]

#set_url = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/asvco2_gas_validation_summary_mirror.csv'

urls = [{'label': 'Summary Mirror', 'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/asvco2_gas_validation_summary_mirror.csv'}]

custom_sets = [{'label': 'EPOFF & APOFF vs Gas Concentration',       'value': 'resids'},
               {'label': 'ZPCAL & SPPCAL vs Ref Gas Concentration',  'value': 'cals'},
               {'label': 'CO2 AVG & STDDEV',                        'value': 'temp resids'},
               {'label': 'CO2 Pres. Mean',                          'value': 'stddev'},
               {'label': 'Residual vs Time',                           'value': 'resid stddev'},
               {'label': 'Residual Histogram',                        'value': 'stddev hist'},
               {'label': 'Summary Table',                           'value': 'summary table'}]

#dataset = data_import.Dataset(urls[0]['value'])

colors = {'Dark': '#111111', 'Light': '#443633', 'text': '#7FDBFF'}

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                requests_pathname_prefix='/co2/validation/',
                external_stylesheets=[dbc.themes.SLATE])
server = app.server

filter_card = dbc.Card(
    dbc.CardBody(
        id='filter_card',
        style={'backgroundColor': colors['Dark']},
        children=[dcc.Checklist(id='filter1'),
                  dcc.Checklist(id='filter2'),
                  dcc.Checklist(id='filter3'),
                  dcc.Checklist(id='filter4'),
                  dhtml.Button(id='update')
        ]
    )

)

tools_card = dbc.Card([
    dbc.CardBody(
           style={'backgroundColor': colors['Dark']},
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
                value='resids',
                clearable=False
                )
            # dash_table.DataTable(
            #     id='datatable',
            #     #data=dataset.to_dict('records'),
            #     # columns=[{'name': 'Serial Number', 'id': 'serial'},
            #     #          {'name': 'Size', 'id': 'size'},
            #     #          {'name': 'State', 'id': 'state'}]
            #     )
    ])
])

graph_card = dbc.Card(
    dbc.CardBody(
        id='display-card',
        children=dcc.Loading(dcc.Graph(id='graphs'))
    )
)


app.layout = dhtml.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([dhtml.H1('ASVCO2 Validation Set')]),
            dbc.Row([
                dbc.Col(children=[tools_card,
                                  dhtml.H5('Filters'),
                                  filter_card,
                                  dcc.RadioItems(id='image_mode',
                                                 options=['Dark', 'Light'],
                                                 value='Dark')
                                  ],
                        width=3),
                dbc.Col(children=graph_card, width=9)
            ])
        ])
    )
])

'''
========================================================================================================================
Callbacks
'''

# plot updating selection
@app.callback(
    [Output('display-card', 'children'),
     Output('filter_card', 'children')],
    [Input('select_set', 'value'),
     Input('select_display', 'value'),
     Input('image_mode', 'value'),
     Input('update', 'n_clicks')],
    [State('filter1', 'value'),
     State('filter2', 'value'),
     State('filter3', 'value'),
     State('filter4', 'value')
     ])

def load_plot(plot_set, plot_fig, im_mode, update, filt1, filt2, filt3, filt4):

    def off_ref(dset, update):
        '''
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
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'OUT_OF_RANGE', 'CO2_DRY_RESIDUAL_REF_LAB_TAG'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['EPOFF & APOFF Residual vs. Reference'],
                                   shared_yaxes=False, shared_xaxes=True)

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        print(changed_id)
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == []:

                return dcc.Graph(figure=load_plots), filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['OUT_OF_RANGE'] == var])
            for var in filt2:
                temp.append(df[df['CO2_DRY_RESIDUAL_REF_LAB_TAG'] == var])

            df = pd.concat(temp)

        # if we are just changing pages, then we need to refresh the filter card
        else:

            filt_list2 = []
            for var in list(df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique()):
                filt_list2.append({'label': var, 'value': var})

            # default filter card
            filt_card = [dhtml.Label('Out of Range'),
                         dcc.Checklist(id='filter1', options=[0, 1], value=[0, 1]),
                         dhtml.Label('Reference Range'),
                         dcc.Dropdown(id='filter2', options=filt_list2, value=df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique()[0],
                                      multi=True, clearable=True, persistence=True),
                         dcc.Checklist(id='filter3'),
                         dcc.Checklist(id='filter4'),
                         dhtml.Button('Update Filter', id='update')]


        epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']
        apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']

        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_RESIDUAL_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='EPOFF Dry TCORR', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_RESIDUAL_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='APOFF Dry TCORR', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='Residual',
                                    xaxis_title='Reference Gas Concentration'
                                    )

        return dcc.Graph(figure=load_plots), filt_card


    def cal_ref(dset, filt1, filt2, filt3, filt4):
        '''
        TODO:


        Select serial and date
            serial: SN_ASVCO2
            date: time
        Residual for ZPPCAL and SPPCAL vs gas concentration
            state: INSTRUMENT_STATE
            gas:
        :return:
        '''

        filt_card = [dhtml.Label(''),
                     dcc.Checklist(id='filter1'),
                     dcc.Checklist(id='filter2'),
                     dcc.Checklist(id='filter3'),
                     dcc.Checklist(id='filter4')]

        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'])

        zcal = df[df['INSTRUMENT_STATE'] == 'ZPPCAL']
        scal = df[df['INSTRUMENT_STATE'] == 'SPPCAL']

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Pressure'],
                                   shared_yaxes=False)

        load_plots.add_scatter(x=zcal['CO2_REF_LAB'], y=zcal['CO2_RESIDUAL_MEAN_ASVCO2'], name='ZPPCAL',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=scal['CO2_REF_LAB'], y=scal['CO2_RESIDUAL_MEAN_ASVCO2'], name='SPPCAL',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=zcal['CO2_REF_LAB'], y=zcal['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='ZPPCAL',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=scal['CO2_REF_LAB'], y=scal['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='SPPCAL',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(
                                    yaxis_title='Residual',
                                    xaxis_title='CO2 Gas Concentration',
                                    )

        return load_plots, filt_card


    def multi_ref(dset, filt1, filt2, filt3, filt4):
        '''
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Residual
        :return:
        TODO:
            Add VALIDATION_date
            Check axes

        '''


        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2',
                            'CO2_RESIDUAL_STDDEV_ASVCO2', 'CO2_STDDEV_ASVCO2', 'SN_ASVCO2', 'ASVCO2_firmware', 'CO2DETECTOR_firmware'])

        # filter block
        if 'filter' in str(dash.callback_context.triggered[0]['prop_id'].split('.')[0]):

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == []:
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return load_plots, filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['SN_ASVCO2'] == var])
            for var in filt2:
                temp.append(df[df['CO2DETECTOR_firmware'] == var])
            for var in filt3:
                temp.append(df[df['ASVCO2_firmware'] == var])

            df = pd.concat(temp)

        else:
            filt_card = [dhtml.Label('Serial #'),
                         dcc.Checklist(id='filter1', options=list(df['SN_ASVCO2'].unique()),
                                     value=list(df['SN_ASVCO2'].unique())),
                         dhtml.Label('LiCOR Firmware'),
                         dcc.Checklist(id='filter2', options=list(df['CO2DETECTOR_firmware'].unique()),
                                     value=list(df['CO2DETECTOR_firmware'].unique())),
                         dhtml.Label('ASVCO2 firmware'),
                         dcc.Checklist(id='filter3', options=list(df['ASVCO2_firmware'].unique()),
                                       value=list(df['ASVCO2_firmware'].unique())),
                         dcc.Checklist(id='filter4')]

        load_plots = make_subplots(rows=2, cols=1,
                                   subplot_titles=['Pressure'],
                                   shared_yaxes=False)

        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_RESIDUAL_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_RESIDUAL_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_RESIDUAL_STDDEV_ASVCO2'],
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=2, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_STDDEV_ASVCO2'],
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=2, col=1)

        load_plots['layout'].update(yaxis1_title='CO2 Residuals',
                                    xaxis1_title='CO2 Gas Concentration',
                                    yaxis2_title='CO2 STDDEV',
                                    xaxis2_title='CO2 Gas Concentration',
                                    )


        return load_plots, filt_card


    def multi_stddev(dset, filt1, filt2, filt3, filt4):
        '''
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Standard deviation
        :return:

        TODO:
            Check axes -- doesn't match hoverdata
            Add date range
            Add Temp corrected
            Check y axes
            Add INSTRUMENT_STATE to hoverinfo
        '''

        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'SN_ASVCO2', 'ASVCO2_firmware', 'CO2DETECTOR_firmware'])

        # filter block
        if 'filter' in str(dash.callback_context.triggered[0]['prop_id'].split('.')[0]):

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == []:
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return load_plots, filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['SN_ASVCO2'] == var])
            for var in filt2:
                temp.append(df[df['CO2DETECTOR_firmware'] == var])
            for var in filt3:
                temp.append(df[df['ASVCO2_firmware'] == var])

            df = pd.concat(temp)

        else:
            filt_card = [dhtml.Label('Serial #'),
                         dcc.Checklist(id='filter1', options=list(df['SN_ASVCO2'].unique()),
                                     value=list(df['SN_ASVCO2'].unique())),
                         dhtml.Label('LiCOR Firmware'),
                         dcc.Checklist(id='filter2', options=list(df['CO2DETECTOR_firmware'].unique()),
                                     value=list(df['CO2DETECTOR_firmware'].unique())),
                         dhtml.Label('ASVCO2 firmware'),
                         dcc.Checklist(id='filter3', options=list(df['ASVCO2_firmware'].unique()),
                                       value=list(df['ASVCO2_firmware'].unique())),
                         dcc.Checklist(id='filter4')]


        zcal = df[df['INSTRUMENT_STATE'] == 'ZPPCAL']
        scal = df[df['INSTRUMENT_STATE'] == 'SPPCAL']

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Pressure'],
                                   shared_yaxes=False)

        load_plots.add_scatter(x=zcal['CO2_REF_LAB'], y=zcal['CO2_RESIDUAL_MEAN_ASVCO2'], name='ZPPCAL',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=scal['CO2_REF_LAB'], y=scal['CO2_RESIDUAL_MEAN_ASVCO2'], name='SPPCAL',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=zcal['CO2_REF_LAB'], y=zcal['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='ZPPCAL',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=scal['CO2_REF_LAB'], y=scal['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='SPPCAL',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(
            yaxis_title='Residual',
            xaxis_title='CO2 Gas Concentration',
        )

        return load_plots, filt_card


    def resid_and_stdev(dset, filt1, filt2, filt3, filt4):
        '''
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Residual vs time, with STDDEV as error bars
        :return:

        TODO:
            Filter by CO2_DRY_RESIDUAL_REF_LAB_TAG

        NOTES:
            At test, the Last Validation filter doesn't appear to work. Upon inspection, this a problem in the data,
            It is a bug in the codxce
        '''

        df = dset.get_data(variables=['CO2_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'SN_ASVCO2', 'ASVCO2_firmware',
                                      'CO2DETECTOR_firmware', 'last_ASVCO2_validation'])

        # filter block
        if 'filter' in str(dash.callback_context.triggered[0]['prop_id'].split('.')[0]):

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == [] or filt4 == []:
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return load_plots, filt_card

            temp = []

            for var in filt1:
                temp.append(df[df['SN_ASVCO2'] == var])
            for var in filt2:
                temp.append(df[df['CO2DETECTOR_firmware'] == var])
            for var in filt3:
                temp.append(df[df['ASVCO2_firmware'] == var])
            for var in filt4:
                temp.append(df[df['last_ASVCO2_validation'] == var])

            df = pd.concat(temp)

        else:
            filt_card = [dhtml.Label('Serial #'),
                         dcc.Checklist(id='filter1', options=list(df['SN_ASVCO2'].unique()),
                                       value=list(df['SN_ASVCO2'].unique())),
                         dhtml.Label('LiCOR Firmware'),
                         dcc.Checklist(id='filter2', options=list(df['CO2DETECTOR_firmware'].unique()),
                                       value=list(df['CO2DETECTOR_firmware'].unique())),
                         dhtml.Label('ASVCO2 firmware'),
                         dcc.Checklist(id='filter3', options=list(df['ASVCO2_firmware'].unique()),
                                       value=list(df['ASVCO2_firmware'].unique())),
                         dhtml.Label('Last Validation'),
                         dcc.Checklist(id='filter4', options=list(df['last_ASVCO2_validation'].unique()),
                                       value=list(df['last_ASVCO2_validation'].unique()))]

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Residual Over Time'],
                                   shared_yaxes=False)

        load_plots.add_scatter(x=df['time'], y=df['CO2_RESIDUAL_MEAN_ASVCO2'], name='Residual',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='Dry Residual',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(
            yaxis_title='Residual'
        )

        return load_plots, filt_card

    def stddev_hist(dset, filt1, filt2, filt3, filt4):
        '''
        Select random variable
        Histogram of marginal probability dists

        TODO:
            Set x-axis max to +/-50
            x-axis bins or at least lines should be in 1ppm increments
            Can we filter by CO2_DRY_RESIDUAL_REF_LAB_TAG, there should be standard ranges on ERDDAP.
            Add filter by last_VALIDATION_DATE (or whatever)

        :return:
        '''

        resid_sets = ['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2']

        df = dset.get_data(variables=['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'INSTRUMENT_STATE'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Residual'],
                                   shared_yaxes=False)

        # filter block
        if 'filter' in str(dash.callback_context.triggered[0]['prop_id'].split('.')[0]):

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == []:
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return load_plots, filt_card

            temp = []

            for var in filt2:
                temp.append(df[df['INSTRUMENT_STATE'] == var])

            df = pd.concat(temp)

            for co2_set in filt1:
                load_plots.add_trace(go.Histogram(x=df[co2_set]), row=1, col=1)

        else:
            filt_card = [dhtml.Label('Residual Type'),
                         dcc.Checklist(id='filter1', options=resid_sets,
                                       value=resid_sets),
                         dhtml.Label('Instrument State'),
                         dcc.Checklist(id='filter2', options=['APOFF', 'EPOFF'],
                                       value=['APOFF', 'EPOFF']),
                         dhtml.Label(''),
                         dcc.Checklist(id='filter3'),
                         dhtml.Label(''),
                         dcc.Checklist(id='filter4')]

            for co2_set in resid_sets:
                load_plots.add_trace(go.Histogram(x=df[co2_set]), row=1, col=1)

        load_plots['layout'].update(
            yaxis_title='Residuals'
        )

        return load_plots, filt_card


    def summary_table(dset, filt1, filt2, filt3, filt4):
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

        # filter block
        if 'filter' in str(dash.callback_context.triggered[0]['prop_id'].split('.')[0]):

            filt_card = dash.no_update

            # if we're filtering everything, plotting is unnecessary
            if filt1 == [] :
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return load_plots, filt_card

            temp = []

            for var in filt2:
                temp.append(df[df['INSTRUMENT_STATE'] == var])

            df = pd.concat(temp)

            for co2_set in filt1:
                load_plots.add_trace(go.Histogram(x=df[co2_set]), row=1, col=1)


        return


    def switch_plot(case):
        return {'resids':        off_ref,
                'cals':          cal_ref,
                'temp resids':   multi_ref,
                'stddev':        multi_stddev,
                'resid stddev':  resid_and_stdev,
                'stddev hist':   stddev_hist,
                'summary table': summary_table
                }.get(case)

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

    dataset = data_import.Dataset(plot_set)
    plotters = switch_plot(plot_fig)(dataset, update)

    if plot_fig == 'summary table':
        pass

    else:
        plotters[0].figure.update_layout(height=600,
            title=' ',
            hovermode='x unified',
            xaxis_showticklabels=True,
            yaxis_fixedrange=True,
            plot_bgcolor=colors[im_mode],
            paper_bgcolor=colors[im_mode],
            font_color=colors['text'],
            autosize=True,
            xaxis=dict(showgrid=False),
            showlegend=True, modebar={'orientation': 'h'},
            margin=dict(l=25, r=25, b=25, t=25, pad=4)
        )

    return plotters[0], plotters[1]


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)


    # # table updating selection
    # @app.callback(
    #     [Output('datatable', 'data'),
    #      Output('datatable', 'columns')],
    #     [Input('select_x', 'value')])
    # def load_plot(selected_set):
    #     def off_ref(df):
    #
    #         # epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']
    #         # apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']
    #
    #         data_set = data_import()
    #
    #         # #dtable = dash_table.DataTable()
    #         #
    #         # columns = [{'id': 'state', 'name': 'State'},
    #         #          {'id': 'size', 'name': 'Size'}]
    #         #
    #         # table_df = pd.concat([epoff, apoff])
    #         # drivers = [list(table_df['INSTRUMENT_STATE'].unique()), list(table_df.groupby('INSTRUMENT_STATE').size())]
    #         # sizes = [str(x[0]) + ", " + str(x[1]) for x in drivers]
    #         #
    #         # table_data = [{'state': list(table_df['INSTRUMENT_STATE'].unique())},
    #         #                      {'size': sizes}]
    #
    #         return  # load_plots#, table_data, columns
    #
    #     def cal_ref(df):
    #         '''
    #         Select serial and date
    #             serial: SN_ASVCO2
    #             date: time
    #         Residual for ZPPCAL and SPPCAL vs gas concentration
    #             state: INSTRUMENT_STATE
    #             gas:
    #         :return:
    #         '''
    #
    #         # test function
    #
    #         epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']
    #         apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']
    #
    #         # #dtable = dash_table.DataTable()
    #         #
    #         # columns = [{'id': 'state', 'name': 'State'},
    #         #          {'id': 'size', 'name': 'Size'}]
    #         #
    #         # table_df = pd.concat([epoff, apoff])
    #         # drivers = [list(table_df['INSTRUMENT_STATE'].unique()), list(table_df.groupby('INSTRUMENT_STATE').size())]
    #         # sizes = [str(x[0]) + ", " + str(x[1]) for x in drivers]
    #         #
    #         # table_data = [{'state': list(table_df['INSTRUMENT_STATE'].unique())},
    #         #                      {'size': sizes}]
    #
    #         return  # load_plots  # , table_data, columns
    #
    #         # return
    #
    #     def multi_ref(df):
    #         '''
    #         Select serial, LICOR firmware, ASVCO2 firmware, date range
    #         Boolean temperature correct residual
    #         Residual
    #         :return:
    #         '''
    #
    #         epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']
    #         apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']
    #
    #         # #dtable = dash_table.DataTable()
    #         #
    #         # columns = [{'id': 'state', 'name': 'State'},
    #         #          {'id': 'size', 'name': 'Size'}]
    #         #
    #         # table_df = pd.concat([epoff, apoff])
    #         # drivers = [list(table_df['INSTRUMENT_STATE'].unique()), list(table_df.groupby('INSTRUMENT_STATE').size())]
    #         # sizes = [str(x[0]) + ", " + str(x[1]) for x in drivers]
    #         #
    #         # table_data = [{'state': list(table_df['INSTRUMENT_STATE'].unique())},
    #         #                      {'size': sizes}]
    #
    #         return  # load_plots  # , table_data, columns
    #
    #     def multi_stddev(df):
    #         '''
    #         Select serial, LICOR firmware, ASVCO2 firmware, date range
    #         Boolean temperature correct residual
    #         Standard deviation
    #         :return:
    #         '''
    #
    #         return
    #
    #     def resid_and_stdev(df):
    #         '''
    #         Select random variable
    #         Histogram of marginal probability dists
    #         :return:
    #         '''
    #
    #         return
    #
    #     def switch_plot(case, data):
    #         return {'resids': off_ref(data),
    #                 'cals': cal_ref(data),
    #                 'temp resids': multi_ref(data),
    #                 'stddev': multi_stddev(data),
    #                 'resid stddev': resid_and_stdev(data)
    #                 }.get(case)
    #
    #     states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']
    #
    #     data = dataset.ret_data()
    #     # print(plot_fig)
    #     # plotters = switch_plot(plot_fig, data)