'''
========================================================================================================================
Start Dashboard
'''

import datetime
import pandas as pd

import dash
from dash import dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html as dhtml

import data_import

rt_url1 = 'https://data.pmel.noaa.gov/generic/erddap/tabledap/sd_shakedown_collection.csv'
rt_url2 = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1067_2021_post_mission.csv'
rt_url3 = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1030_2021_post_mission.csv'
url4 = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1091_ecmwf_2021.csv'
url5 = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/sd1089_ecmwf_2021.csv'
set_loc = 'D:\Data\CO2 Sensor tests\\asvco2_gas_validation_all_fixed_station_mirror.csv'

sets = [{'label': 'SD Shakedown', 'value': rt_url1},
        #{'label': 'SD 1067', 'value': rt_url2},
        #{'label': 'SD 1030', 'value': rt_url3},
        #{'label': 'SD 1091', 'value': url4},
        {'label': 'SD 1089', 'value': url5}]

custom_sets = [{'label': 'XCO2 Mean', 'value': 'co2_raw'},
        {'label': 'XCO2 Residuals', 'value': 'co2_res'},
        {'label': 'XCO2 Delta', 'value': 'co2_delt'},
        {'label': 'CO2 Pres. Mean', 'value': 'co2_det_state'},
        {'label': 'CO2 Mean', 'value': 'co2_mean_zp'},
        {'label': 'CO2 Mean SP', 'value': 'co2_mean_sp'},
        {'label': 'CO2 Span & Temp', 'value': 'co2_span_temp'},
        {'label': 'CO2 Zero Temp', 'value': 'co2_zero_temp'},
        {'label': 'CO2 STDDEV', 'value': 'co2_stddev'},
        {'label': 'O2 Mean', 'value': 'o2_mean'},
        {'label': 'CO2 Span', 'value': 'co2_span'},
        {'label': 'CO2 Zero', 'value': 'co2_zero'},
        {'label': 'Pres Difference', 'value': 'pres_state'}
        ]


dataset = data_import.Dataset(rt_url1)


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
                start_date=dataset.t_end - datetime.timedelta(days=14),
                end_date=dataset.t_end),
            dhtml.Label(['Select Mission']),
            dcc.Dropdown(
                    id='set-select',
                    options=sets,
                    value=sets[0]['value'],
                    clearable=False
                ),
            dhtml.Label(['Select Plots']),
            dcc.Dropdown(
                    id="select_x",
                    options=custom_sets,
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

#engineering data selection
@app.callback(
    Output('graphs', 'figure'),
    [Input('select_x', 'value'),
     Input('set-select', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('mode-select', 'value')#,
     #Input('graphs', 'figure')
     ])

def plot_evar(selection, set, t_start, t_end, colormode):

    def co2_raw(df):
        '''
        #1
        'co2_raw'
        'XCO2 Mean',
            Primary: XCO2_DRY_SW_MEAN_ASVCO2 & XCO2_DRY_AIR_MEAN_ASVCO2
            Secondary: SSS and SST
        '''

        dataset = data_import.Dataset(set, ['XCO2_DRY_SW_MEAN_ASVCO2',
                                            'XCO2_DRY_AIR_MEAN_ASVCO2',
                                            'SAL_SBE37_MEAN',
                                            'TEMP_SBE37_MEAN'])

        df = dataset.ret_data()

        load_plots = make_subplots(rows=3, cols=1, shared_xaxes='all',
                                   subplot_titles=("XCO2 DRY", "SSS", "SST"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=df['time'], y=df['XCO2_DRY_SW_MEAN_ASVCO2'].dropna(), mode='markers',
                               marker={'size': 2}, name='Seawater CO2', hoverinfo='x+y+name', row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['XCO2_DRY_AIR_MEAN_ASVCO2'], mode='markers',
                               marker={'size': 2}, name='CO2 Air', hoverinfo='x+y+name', row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['SAL_SBE37_MEAN'].dropna(), mode='markers',
                               marker={'size': 2}, name='SSS', hoverinfo='x+y+name', row=2, col=1)
        load_plots.add_scatter(x=df['time'], y=df['TEMP_SBE37_MEAN'].dropna(), mode='markers',
                               marker={'size': 2}, name='SST', hoverinfo='x+y+name', row=3, col=1)


        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True, xaxis3_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True, yaxis3_fixedrange=True,
                                    yaxis_title='Dry CO2',
                                    yaxis2_title='Salinity', yaxis3_title='SW Temp',
                                    showlegend=False, modebar={'orientation': 'h'},
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_res(df):
        '''
        #2
        'co2_res'
        'XCO2 Residuals',
            Primary: Calculate residual of XCO2_DRY_SW_MEAN_ASVCO2 - XCO2_DRY_AIR_MEAN_ASVCO2
            Secondary: O2_SAT_SBE37_MEAN and/or O2_MEAN_ASVCO2
            '''
        co2_diff = []

        templist1 = df['XCO2_DRY_SW_MEAN_ASVCO2'].to_list()
        templist2 = df['XCO2_DRY_AIR_MEAN_ASVCO2'].to_list()

        for n in range(len(templist1)):

            co2_diff.append(templist1[n] - templist2[n])


        # co2_diff = go.Scatter(x=df['time'], y=co2_diff, name='CO2 Diff', hoverinfo='x+y+name')
        # sbe = go.Scatter(x=df['time'], y=df['O2_SAT_SBE37_MEAN'], name='O2_SAT_SBE37_MEAN', hoverinfo='x+y+name')
        # o2 = go.Scatter(x=df['time'], y=df['O2_MEAN_ASVCO2'], name='O2_MEAN_ASVCO2', hoverinfo='x+y+name')

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


        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True, xaxis3_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True, yaxis3_fixedrange=True,
                                    yaxis_title='CO2 Diff (Dry-Air)',
                                    yaxis2_title='O2 Mean', yaxis3_title='O2 Mean',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_delt(df):
        '''
        #3
        'co2_delt'
        'XCO2 Delta',
            Primary: calculated pressure differentials between like states
        '''

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

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='Pressure Mean',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_det_state(df):
        '''
        #4
        'co2_det_state'
        'CO2 Pres. Mean',
            Primary: CO2DETECTOR_PRESS_MEAN_ASVCO2 for each state
        '''

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

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='CO2 Mean Pressure',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_mean_zp(df):
        '''
        #5
        'co2_mean_zp'
        'CO2 Mean',
            Primary: CO2_MEAN_ASVCO2 for ZPON, ZPOFF and ZPPCAL
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

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


        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='CO2 Mean',
                                    yaxis2_title='Temp. Mean',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_mean_sp(df):
        '''
        #6
        'co2_mean_sp'
        'CO2 Mean SP',
            Primary: CO2_MEAN_ASVCO2 for SPON, SPOFF, SPPCAL
            Secondary: CO2_MEAN_ASVCO2 SPOFF
        '''

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


        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='CO2 Mean',
                                    yaxis2_title='Temp Mean',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_span_temp(df):
        '''
        #7
        'co2_span_temp'
        'CO2 Span & Temp'
            Primary: CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2 vs. SPOFF CO2DETECTOR_TEMP_MEAN_ASVCO2
        '''

        dset = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        co2 = go.Scatter()

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=['SPOFF Temp vs Span'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=dset['CO2DETECTOR_TEMP_MEAN_ASVCO2'], y=dset['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'],
                               name='CO2 Detector', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis_title='Temp Mean',
                                    yaxis_fixedrange=True,
                                    yaxis_title='Span Coefficient',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_zero_temp(df):
        '''
        #8
        'co2_zero_temp'
        'CO2 Zero Temp',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2 vs. ZPOFF CO2DETECTOR_TEMP_MEAN_ASVCO2
        '''
        dset = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=['Zero Coefficient'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=dset['CO2DETECTOR_TEMP_MEAN_ASVCO2'], y=dset['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'],
                               name='CO2 Detector', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    xaxis_title='Temp Mean',
                                    yaxis_title='Span Coefficient',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin = dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_stddev(df):
        '''
        #9
        'co2_stddev'
        'CO2 STDDEV',
            Primary: CO2_STDDEV_ASVCO2
        '''

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=('CO2 Standard Deviation'),
                                   shared_yaxes=False, vertical_spacing=0.1)

        for n in range(1, len(states)):
            cur_state = df[df['INSTRUMENT_STATE'] == states[n]]
            load_plots.add_scatter(x=cur_state['time'], y=cur_state['CO2_STDDEV_ASVCO2'], name=states[n], hoverinfo='x+y+name',
                                   mode='markers', marker={'size': 2}, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='CO2_STDDEV_ASVCO2',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin = dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def o2_mean(df):
        '''
        #10
        'o2_mean'
        'O2 Mean',
            Primary: O2_MEAN_ASVCO2 for APOFF and EPOFF
        '''

        apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']
        epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all', subplot_titles=['O2 - APOFF', 'O2 -  EPOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=apoff['time'], y=apoff['O2_MEAN_ASVCO2'], name='SPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=epoff['time'], y=epoff['O2_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=2, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='SPOFF',
                                    yaxis2_title='EPOFF',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin = dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_span(df):
        '''
        #11
        'co2_span'
        'CO2 Span',
            Primary: CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for SPOFF
        '''

        dset = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all', subplot_titles=['Span Coeffient', 'Mean Temperature - SPOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=df['time'], y=df['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2'],
                               name='CO2 Span Coef.', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=dset['time'], y=df['CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                               name='Temp Mean', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=2, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='Span Coef',
                                    yaxis2_title='Temp Mean',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin = dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def co2_zero(df):
        '''
        #12
        'co2_zero'
        'CO2 Zero',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

        dset = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all', subplot_titles=['Zero Span Coefficient', 'Temperature Mean - ZPOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=df['time'], y=df['CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2'],
                               name='CO2 Span Coef.', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=1, col=1)
        load_plots.add_scatter(x=dset['time'], y=df['CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                               name='Temp Mean', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 2}, row=2, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    xaxis2_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='Span Coef',
                                    yaxis2_title='Temp Mean',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin = dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def pres_state(df):
        '''
        #13
        'pres_state'
        'Pressure State',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

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

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='Pres. Diff.',
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        return load_plots


    def switch_plot(case, data):
        return {'co2_raw':      co2_raw(data),
        'co2_res':          co2_res(data),
        'co2_delt':         co2_delt(data),
        'co2_det_state':    co2_det_state(data),
        'co2_mean_zp':      co2_mean_zp(data),
        'co2_mean_sp':      co2_mean_sp(data),
        'co2_span_temp':    co2_span_temp(data),
        'co2_zero_temp':    co2_zero_temp(data),
        'co2_stddev':       co2_stddev(data),
        'o2_mean':          o2_mean(data),
        'co2_span':         co2_span(data),
        'co2_zero':         co2_zero(data),
        'pres_state':       pres_state(data)
        }.get(case)

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

    dataset = data_import.Dataset(set)
    data = dataset.ret_data(t_start=t_start, t_end=t_end)

    plotters = switch_plot(selection, data)

    bkgrd_colors = {'Dark': 'background',
                    'Light': 'light'}

    plotters.update_layout(
         plot_bgcolor=colors[bkgrd_colors[colormode]],
         paper_bgcolor=colors['background'],
         font_color=colors['text'],
         autosize=True
    )

    return plotters



if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)
