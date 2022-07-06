'''
========================================================================================================================
Start Dashboard

TODO:
    Dashboard is running painfully slow, it looks like the datasets are huge
        Maybe give the user the ability to decimate and control the rate?

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

# rt_url1 = 'https://data.pmel.noaa.gov/generic/erddap/tabledap/sd_shakedown_collection.csv'
# rt_url2 = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1067_2021_post_mission.csv'
# rt_url3 = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1030_2021_post_mission.csv'
# url4 = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1091_ecmwf_2021.csv'
# url5 = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1089_ecmwf_2021.csv'
# set_loc = 'D:\Data\CO2 Sensor tests\\asvco2_gas_validation_all_fixed_station_mirror.csv'

available_sets = [{'label': 'SD Shakedown',   'value': 'https://data.pmel.noaa.gov/generic/erddap/tabledap/sd_shakedown_collection.csv'},
        {'label': 'SD 1067',        'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1067_2021_post_mission.csv'},
        {'label': 'SD 1030',        'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1030_2021_post_mission.csv'},
#         {'label': 'SD 1091 ECMWF',  'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1091_ecmwf_2021.csv'},
#         {'label': 'SD 1089 ECMWF',  'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1089_ecmwf_2021.csv'},
        {'label': 'SD 1033',        'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1033_tpos_2022.csv'},
        {'label': 'SD 1052 TPOS',   'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1052_tpos_2022.csv'}]

custom_sets = [{'label': 'XCO2 Mean',   'value': 'co2_raw'},
        {'label': 'XCO2 Residuals',     'value': 'co2_res'},
        {'label': 'XCO2 Delta',         'value': 'co2_delt'},
        {'label': 'CO2 Pres. Mean',     'value': 'co2_det_state'},
        {'label': 'CO2 Mean',           'value': 'co2_mean_zp'},
        {'label': 'CO2 Mean SP',        'value': 'co2_mean_sp'},
        {'label': 'CO2 Span & Temp',    'value': 'co2_span_temp'},
        {'label': 'CO2 Zero Temp',      'value': 'co2_zero_temp'},
        {'label': 'CO2 STDDEV',         'value': 'co2_stddev'},
        {'label': 'O2 Mean',            'value': 'o2_mean'},
        {'label': 'CO2 Span',           'value': 'co2_span'},
        {'label': 'CO2 Zero',           'value': 'co2_zero'},
        {'label': 'Pres Difference',    'value': 'pres_state'}
        ]

dataset = data_import.Dataset(available_sets[0]['value'])


colors = {'background': '#111111', 'text': '#7FDBFF', 'light': '#7f7f7f'}

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                requests_pathname_prefix='/co2/real_time/',
                external_stylesheets=[dbc.themes.SLATE])
server = app.server

tools_card = dbc.Card(
    dbc.CardBody(
            style={'backgroundColor': colors['background']},
            children=[dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=dataset.t_start,
                max_date_allowed=dataset.t_end,
                start_date=dataset.t_end - datetime.timedelta(days=7),
                end_date=dataset.t_end),
            dhtml.Label(['Select Mission']),
            dcc.Dropdown(
                    id='set-select',
                    options=available_sets,
                    value=available_sets[0]['value'],
                    clearable=False
                ),
            dhtml.Label(['Select Plots']),
            dcc.Dropdown(
                    id="display-select",
                    options=custom_sets,
                    #value=custom_sets[0]['value'],
                    value='co2_raw',
                    clearable=False
            )
        ])
)

graph_card = dbc.Card(
    [dbc.CardBody(
         [dcc.Loading(dcc.Graph(id='graphs'))]
        )
    ]
)


app.layout = dhtml.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([dhtml.H1('ASVCO2 Real Time')]),
            dbc.Row([
                dbc.Col(children=[
                        tools_card,
                        dcc.RadioItems(id='mode-select',
                           options=[{'label': 'Dark', 'value': 'Dark'},
                                    {'label': 'Light', 'value': 'Light'}],
                           value='Dark',
                           persistence=True)],
                        width=3),
                dbc.Col(graph_card, width=9)
            ])
        ])
    )
])

'''
========================================================================================================================
Callbacks
'''

#update date limits when switching sets
@app.callback(
    [Output('date-picker', 'start_date'),
    Output('date-picker', 'end_date'),
    Output('date-picker', 'max_date_allowed'),
    Output('date-picker', 'min_date_allowed')],
    Input('set-select', 'value'))

def change_set(dataset_url):

    dataset = data_import.Dataset(dataset_url)

    min_date_allowed = dataset.t_start.date()
    max_date_allowed = dataset.t_end.date()
    end_date = dataset.t_end.date()
    start_date = end_date - datetime.timedelta(days=7)

    return start_date, end_date, min_date_allowed, max_date_allowed

#engineering data selection
@app.callback(
    Output('graphs', 'figure'),
    [Input('display-select', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('mode-select', 'value')],
    State('set-select', 'value'))

def plot_evar(selection, t_start, t_end, colormode, erddap_set):

    def co2_raw(dataset):
        '''
        #1
        'co2_raw'
        'XCO2 Mean',
            Primary: XCO2_DRY_SW_MEAN_ASVCO2 & XCO2_DRY_AIR_MEAN_ASVCO2
            Secondary: SSS and SST
        '''

        df = dataset.get_data(variables=['XCO2_DRY_SW_MEAN_ASVCO2', 'XCO2_DRY_AIR_MEAN_ASVCO2', 'SAL_SBE37_MEAN', 'TEMP_SBE37_MEAN'],
                              window_start=t_start, window_end=t_end)

        load_plots = make_subplots(rows=3, cols=1, shared_xaxes='all',
                                   subplot_titles=("XCO2 DRY", "SSS", "SST"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        # customdata = list(zip(df[filt_cols[0]], df[filt_cols[1]], df[filt_cols[2]], df[filt_cols[3]]))
        #
        # hovertemplate = f'CO2 Reference: %{{x}}<br>Residual: %{{y}} <br> {filt_cols[0]}: %{{customdata[0]}}<br>' \
        #                 f'{filt_cols[1]}: %{{customdata[1]}} <br> {filt_cols[2]}: %{{customdata[2]}}<br>' \
        #                 f'{filt_cols[3]}: %{{customdata[3]}}'


        load_plots.add_scatter(x=df['time'], y=df['XCO2_DRY_SW_MEAN_ASVCO2'], mode='markers',
                               marker={'size': 3}, name='Seawater CO2', hoverinfo='x+y+name', row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['XCO2_DRY_AIR_MEAN_ASVCO2'], mode='markers',
                               marker={'size': 3}, name='CO2 Air', hoverinfo='x+y+name', row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['SAL_SBE37_MEAN'], mode='markers',
                               marker={'size': 2}, name='SSS', hoverinfo='x+y+name', row=2, col=1)
        load_plots.add_scatter(x=df['time'], y=df['TEMP_SBE37_MEAN'], mode='markers',
                               marker={'size': 2}, name='SST', hoverinfo='x+y+name', row=3, col=1)


        load_plots['layout'].update(
                                    xaxis2_showticklabels=True, xaxis3_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True, yaxis3_fixedrange=True,
                                    yaxis_title='Dry CO2',
                                    yaxis2_title='Salinity', yaxis3_title='SW Temp',
                                    )

        return load_plots


    def co2_res(dataset):
        '''
        #2
        'co2_res'
        'XCO2 Residuals',
            Primary: Calculate residual of XCO2_DRY_SW_MEAN_ASVCO2 - XCO2_DRY_AIR_MEAN_ASVCO2
            Secondary: O2_SAT_SBE37_MEAN and/or O2_MEAN_ASVCO2
            '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'XCO2_DRY_SW_MEAN_ASVCO2', 'XCO2_DRY_AIR_MEAN_ASVCO2', 'O2_SAT_SBE37_MEAN', 'O2_MEAN_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        co2_diff = []

        templist1 = df['XCO2_DRY_SW_MEAN_ASVCO2'].to_list()
        templist2 = df['XCO2_DRY_AIR_MEAN_ASVCO2'].to_list()

        for n in range(len(templist1)):

            co2_diff.append(templist1[n] - templist2[n])

        load_plots = make_subplots(rows=3, cols=1, shared_xaxes='all',
                                   subplot_titles=("XCO2 DRY-AIR", "Mean O2 SBE37", "Mean O2 ASVCO2"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=df['time'], y=co2_diff, name='CO2 Diff', hoverinfo='x+y+name',
                                mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['O2_SAT_SBE37_MEAN'], name='O2_SAT_SBE37_MEAN', hoverinfo='x+y+name',
                                mode='markers', marker={'size': 2}, row=2, col=1)

        for n in range(1, len(states)):
            cur_state = df[df['INSTRUMENT_STATE'] == states[n]]
            load_plots.add_scatter(x=cur_state['time'], y=cur_state['O2_MEAN_ASVCO2'], name=states[n], hoverinfo='x+y+name',
                                    mode='markers', marker={'size': 2},  row=3, col=1)


        load_plots['layout'].update(xaxis2_showticklabels=True, xaxis3_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True, yaxis3_fixedrange=True,
                                    yaxis_title='CO2 Diff (Dry-Air)',
                                    yaxis2_title='O2 Mean', yaxis3_title='O2 Mean'
                                    )

        return load_plots


    def co2_delt(dataset):
        '''
        #3
        'co2_delt'
        'XCO2 Delta',
            Primary: calculated pressure differentials between like states
        '''

        df = dataset.get_data(variables=['CO2DETECTOR_PRESS_MEAN_ASVCO2',
                                         'INSTRUMENT_STATE'],
                              window_start=t_start, window_end=t_end)

        df.dropna(axis='rows', subset=['CO2DETECTOR_PRESS_MEAN_ASVCO2'], inplace=True)

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Pressure'],
                                   shared_yaxes=False)

        temp1 = df[df['INSTRUMENT_STATE'] == 'ZPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):

            co2_diff.append(templist1[n] - templist2[n])

        load_plots.add_scatter(x=temp1['time'], y=co2_diff, name='ZP', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'SPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        load_plots.add_scatter(x=temp1['time'], y=co2_diff, name='SP', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'EPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'EPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        load_plots.add_scatter(x=temp1['time'], y=co2_diff, name='EP', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'APON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'APOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        load_plots.add_scatter(x=temp1['time'], y=co2_diff, name='AP', hoverinfo='x+y+name',
                             mode='markers', marker={'size': 2}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='Pressure Mean')

        return load_plots


    def co2_det_state(dataset):
        '''
        #4
        'co2_det_state'
        'CO2 Pres. Mean',
            Primary: CO2DETECTOR_PRESS_MEAN_ASVCO2 for each state
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_PRESS_MEAN_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        temp = df[df['INSTRUMENT_STATE'] == 'ZPON']

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Pressure'],
                                   shared_yaxes=False)

        load_plots.add_scatter(x=temp['time'], y=temp['CO2DETECTOR_PRESS_MEAN_ASVCO2'], name='ZPON', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        for n in range(1, len(states)):

             cur_state = df[df['INSTRUMENT_STATE'] == states[n]]

             load_plots.add_scatter(x=cur_state['time'], y=cur_state['CO2DETECTOR_PRESS_MEAN_ASVCO2'],
                                     name=states[n], hoverinfo='x+y+name',
                                     mode='markers', marker={'size': 2},  row=1, col=1)

        load_plots['layout'].update(yaxis_title='CO2 Mean Pressure')

        return load_plots


    def co2_mean_zp(dataset):
        '''
        #5
        'co2_mean_zp'
        'CO2 Mean',
            Primary: CO2_MEAN_ASVCO2 for ZPON, ZPOFF and ZPPCAL
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2_MEAN_ASVCO2', "CO2DETECTOR_TEMP_MEAN_ASVCO2"],
                              window_start=t_start, window_end=t_end)

        primary1 = df[df['INSTRUMENT_STATE'] == 'ZPON']
        primary2 = df[df['INSTRUMENT_STATE'] == 'ZPOFF']
        primary3 = df[df['INSTRUMENT_STATE'] == 'ZPPCAL']
        secondary = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all',
                                   subplot_titles=("CO2_MEAN_ASVCO2", "CO2DETECTOR_TEMP_MEAN_ASVCO2"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=primary1['time'], y=primary1['CO2_MEAN_ASVCO2'], name='ZPON', hoverinfo='x+y+name',
                                mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=primary2['time'], y=primary2['CO2_MEAN_ASVCO2'], name='ZPOFF', hoverinfo='x+y+name',
                                mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=primary3['time'], y=primary3['CO2_MEAN_ASVCO2'], name='ZPPCAL', hoverinfo='x+y+name',
                                mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=secondary['time'], y=secondary['CO2DETECTOR_TEMP_MEAN_ASVCO2'].dropna(),
                                name='ZPOFF', hoverinfo='x+y+name',
                                mode='markers', marker={'size': 2}, row=2, col=1)


        load_plots['layout'].update(xaxis2_showticklabels=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='CO2 Mean',
                                    yaxis2_title='Temp. Mean',
                                    )

        return load_plots


    def co2_mean_sp(dataset):
        '''
        #6
        'co2_mean_sp'
        'CO2 Mean SP',
            Primary: CO2_MEAN_ASVCO2 for SPON, SPOFF, SPPCAL
            Secondary: CO2_MEAN_ASVCO2 SPOFF
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', "CO2_MEAN_ASVCO2", "CO2DETECTOR_TEMP_MEAN_ASVCO2"],
                              window_start=t_start, window_end=t_end)

        primary1 = df[df['INSTRUMENT_STATE'] == 'SPON']
        primary2 = df[df['INSTRUMENT_STATE'] == 'SPOFF']
        primary3 = df[df['INSTRUMENT_STATE'] == 'SPPCAL']
        secondary = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all',
                                   subplot_titles=("CO2_MEAN_ASVCO2", "CO2DETECTOR_TEMP_MEAN_ASVCO2"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=primary1['time'], y=primary1['CO2_MEAN_ASVCO2'], name='SPON', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=primary2['time'], y=primary2['CO2_MEAN_ASVCO2'], name='SPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=primary3['time'], y=primary3['CO2_MEAN_ASVCO2'], name='SPPCAL', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=secondary['time'], y=secondary['CO2DETECTOR_TEMP_MEAN_ASVCO2'].dropna(), name='ZPOFF',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=2, col=1)


        load_plots['layout'].update(xaxis2_showticklabels=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='CO2 Mean',
                                    yaxis2_title='Temp Mean'
                                    )

        return load_plots


    def co2_span_temp(dataset):
        '''
        #7
        'co2_span_temp'
        'CO2 Span & Temp'
            Primary: CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2 vs. SPOFF CO2DETECTOR_TEMP_MEAN_ASVCO2
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_TEMP_MEAN_ASVCO2',
                                         'CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        dset = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        #co2 = go.Scatter()

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=['SPOFF Temp vs Span'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=dset['CO2DETECTOR_TEMP_MEAN_ASVCO2'], y=dset['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'],
                               name='CO2 Detector', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        load_plots['layout'].update(xaxis_title='Temp Mean',
                                    yaxis_title='Span Coefficient'
                                    )

        return load_plots


    def co2_zero_temp(dataset):
        '''
        #8
        'co2_zero_temp'
        'CO2 Zero Temp',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2 vs. ZPOFF CO2DETECTOR_TEMP_MEAN_ASVCO2
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_TEMP_MEAN_ASVCO2',
                                         'CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        dset = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=['Zero Coefficient'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=dset['CO2DETECTOR_TEMP_MEAN_ASVCO2'], y=dset['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'],
                               name='CO2 Detector', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        load_plots['layout'].update(xaxis_title='Temp Mean',
                                    yaxis_title='Span Coefficient',
                                    )

        return load_plots


    def co2_stddev(dataset):
        '''
        #9
        'co2_stddev'
        'CO2 STDDEV',
            Primary: CO2_STDDEV_ASVCO2
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2_STDDEV_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=('CO2 Standard Deviation'),
                                   shared_yaxes=False, vertical_spacing=0.1)

        for n in range(1, len(states)):
            cur_state = df[df['INSTRUMENT_STATE'] == states[n]]
            load_plots.add_scatter(x=cur_state['time'], y=cur_state['CO2_STDDEV_ASVCO2'], name=states[n], hoverinfo='x+y+name',
                                   mode='markers', marker={'size': 2}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='CO2_STDDEV_ASVCO2')

        return load_plots


    def o2_mean(dataset):
        '''
        #10
        'o2_mean'
        'O2 Mean',
            Primary: O2_MEAN_ASVCO2 for APOFF and EPOFF
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'O2_MEAN_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']
        epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all', subplot_titles=['O2 - APOFF', 'O2 -  EPOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=apoff['time'], y=apoff['O2_MEAN_ASVCO2'], name='SPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=epoff['time'], y=epoff['O2_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=2, col=1)

        load_plots['layout'].update(yaxis_title='SPOFF',
                                    yaxis2_title='EPOFF',
                                    )

        return load_plots


    def co2_span(dataset):
        '''
        #11
        'co2_span'
        'CO2 Span',
            Primary: CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for SPOFF
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2',
                                         'CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        dset = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all', subplot_titles=['Span Coeffient', 'Mean Temperature - SPOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=df['time'], y=df['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'],
                               name='CO2 Span Coef.', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=dset['time'], y=df['CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                               name='Temp Mean', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=2, col=1)

        load_plots['layout'].update(xaxis2_showticklabels=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='Span Coef',
                                    yaxis2_title='Temp Mean',
                                    )

        return load_plots


    def co2_zero(dataset):
        '''
        #12
        'co2_zero'
        'CO2 Zero',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2',
                                         'CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        dset = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all', subplot_titles=['Zero Span Coefficient', 'Temperature Mean - ZPOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=df['time'], y=df['CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2'],
                               name='CO2 Span Coef.', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=dset['time'], y=df['CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                               name='Temp Mean', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=2, col=1)

        load_plots['layout'].update(xaxis2_showticklabels=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='Span Coef',
                                    yaxis2_title='Temp Mean',
                                    )

        return load_plots


    def pres_state(dataset):
        '''
        #13
        'pres_state'
        'Pressure State',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_PRESS_MEAN_ASVCO2',
                                         'CO2DETECTOR_PRESS_UNCOMP_STDDEV_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        dset = df[df['INSTRUMENT_STATE'] == 'APOFF']

        pres_diff = []

        templist1 = dset['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = dset['CO2DETECTOR_PRESS_UNCOMP_STDDEV_ASVCO2'].to_list()

        for n in range(len(templist1)):

            pres_diff.append(templist1[n] - templist2[n])

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=['Pressure Differential - APOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=dset['time'], y=pres_diff,
                               name='CO2 Span Coef.', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='Pres. Diff.')

        return load_plots

    def switch_plot(case):
        return {'co2_raw':      co2_raw,
        'co2_res':          co2_res,
        'co2_delt':         co2_delt,
        'co2_det_state':    co2_det_state,
        'co2_mean_zp':      co2_mean_zp,
        'co2_mean_sp':      co2_mean_sp,
        'co2_span_temp':    co2_span_temp,
        'co2_zero_temp':    co2_zero_temp,
        'co2_stddev':       co2_stddev,
        'o2_mean':          o2_mean,
        'co2_span':         co2_span,
        'co2_zero':         co2_zero,
        'pres_state':       pres_state
        }.get(case)

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

    dataset = data_import.Dataset(erddap_set)

    plotters = switch_plot(selection)(dataset)

    bkgrd_colors = {'Dark': 'background',
                    'Light': 'light'}

    plotters.update_layout(
         height=600,
         title=' ',
         #hovermode='x unified',
         xaxis_showticklabels=True,
         plot_bgcolor=colors[bkgrd_colors[colormode]],
         paper_bgcolor=colors['background'],
         font_color=colors['text'],
         autosize=True,
         showlegend = False,
         modebar = {'orientation': 'h'},
         margin = dict(l=25, r=25, b=25, t=25, pad=4)
    )

    return plotters



if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)
