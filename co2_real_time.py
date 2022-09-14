'''
========================================================================================================================
Start Dashboard

TODO:
   Maybe give the user the ability to decimate and control the rate?

'''

import datetime
import pandas as pd
import numpy as np

import dash
from dash import dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html as dhtml

import data_import

# useful characters for labels
mark_size = 3
mu = f'\u03BC'
inverse = f'\u207B\u00b9'
squared = f'\u00B2'
sub2 = f'\u2082'

# import datasets from external spreadsheet

available_sets = []

with open('real_time_sets.csv', 'r') as csv:

    for n, line in enumerate(csv.readlines()):

        if n == 0:
            continue

        available_sets.append({'label': line.split(',')[0], 'value': line.split(',')[1].strip()})

# dashboard definitions
custom_sets = [{'label': f'xCO{sub2} / SST / SSS',              'value': 'co2_raw'},
        {'label': f'xCO{sub2} / SBE O{sub2} / ASVCO2 O{sub2}',  'value': 'co2_res'},
        {'label': f'Delta Pressures',                           'value': 'co2_delt'},
        {'label': f'Pump Pressures',                            'value': 'co2_det_state'},
        {'label': f'Zero Position xCO{sub2} / Detector Temp',   'value': 'co2_mean_zp'},
        {'label': f'Span Position xCO{sub2} / Detector Temp',   'value': 'co2_mean_sp'},
        {'label': f'Span Coefficient vs Temp',                  'value': 'co2_span_temp'},
        {'label': f'Zero Coefficient vs Temp',                  'value': 'co2_zero_temp'},
        {'label': f'CO{sub2} STDDEV',                           'value': 'co2_stddev'},
        {'label': f'AVSCO{sub2} O{sub2}',                       'value': 'o2_mean'},
        {'label': f'Span Coefficient Time Series',              'value': 'co2_span'},
        {'label': f'Zero Coefficient Time Series',              'value': 'co2_zero'},
        {'label': f'ASVCO{sub2} Pressure Check',               'value': 'pres_state'}
        ]

# init dataset
dataset = data_import.Dataset(available_sets[1]['value'])

# colors for the plots
colors = {'Dark': {'bckgrd': '#111111', 'text': '#7FDBFF'},
          'Light': {'bckgrd': '#FAF9F6', 'text': '#111111'},
          'Green':  '#2ECC40',
          'Blue':   '#0000FF',
          'Red':    '#FF4136'}


# init dashboard
app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                requests_pathname_prefix='/co2/real_time/',
                external_stylesheets=[dbc.themes.SLATE])
server = app.server

# dashboard layout
tools_card = dbc.Card(
    dbc.CardBody(
            style={'backgroundColor': colors['Dark']['bckgrd']},
            children=[
                dhtml.H5(id='daterange', children='Date Range'),
                dcc.DatePickerRange(
                id='date-picker'
                ),
            dhtml.Label(['Select Mission']),
            dcc.Dropdown(
                    id='set-select',
                    options=available_sets,
                    # value=available_sets[0]['value'],
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
            dbc.Row([dhtml.H1(f'ASVCO{sub2} Real Time')]),
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

def ls_regression(x, y):

    x_array = np.array(x)
    y_array = np.array(y)

    mx = np.ma.masked_invalid(x_array * y_array)

    x_sum = np.sum(x_array)
    y_sum = np.sum(y_array)
    x_sqr = x_array ** 2
    x_sumsqr = np.sum(x_sqr)
    xy_sum = np.sum(x_array * y_array)
    N = len(x_array)

    m = ((N * xy_sum) - (x_sum * y_sum)) / ((N * x_sumsqr) - x_sum ** 2)
    b = (y_sum - (m * x_sum)) / N

    return m * x_array - b, x_array


#update date limits when switching sets
@app.callback(
    [Output('date-picker', 'start_date'),
    Output('date-picker', 'end_date'),
    Output('date-picker', 'max_date_allowed'),
    Output('date-picker', 'min_date_allowed'),
    Output('daterange', 'children')],
    Input('set-select', 'value'))

def change_set(dataset_url):

    if dataset_url is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    dataset = data_import.Dataset(dataset_url)

    max_date_allowed = dataset.t_start.date()
    min_date_allowed = dataset.t_end.date()
    end_date = dataset.t_end.date()
    start_date = end_date - datetime.timedelta(days=7)
    d_range = f'Start: {max_date_allowed} End: {min_date_allowed}'

    return start_date, end_date, min_date_allowed, max_date_allowed, d_range

#engineering data selection
@app.callback(
    Output('graphs', 'figure'),
    [Input('display-select', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('mode-select', 'value')],
    State('set-select', 'value'))

def plot_evar(selection, t_start, t_end, colormode, erddap_set):

    if erddap_set is None:
        return dash.no_update

    def co2_raw(dataset):
        '''
        #1
        'co2_raw'
        'xCO2 / SST / SSS',
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
                               marker={'size': mark_size}, name='Seawater CO2', hoverinfo='x+y+name', row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['XCO2_DRY_AIR_MEAN_ASVCO2'], mode='markers',
                               marker={'size': mark_size}, name='CO2 Air', hoverinfo='x+y+name', row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['SAL_SBE37_MEAN'], mode='markers',
                               marker={'size': mark_size}, name='SSS', hoverinfo='x+y+name', row=2, col=1)
        load_plots.add_scatter(x=df['time'], y=df['TEMP_SBE37_MEAN'], mode='markers',
                               marker={'size': mark_size}, name='SST', hoverinfo='x+y+name', row=3, col=1)


        load_plots['layout'].update(
                                    xaxis2_showticklabels=True, xaxis3_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True, yaxis3_fixedrange=True,
                                    yaxis_title=f'Dry CO2 ({mu} mol{inverse})',
                                    yaxis2_title='Salinity', yaxis3_title='SW Temp (°C)',
                                    showlegend=True,
                                    yaxis2_gridcolor=colors[colormode]['text'],
                                    xaxis2_gridcolor=colors[colormode]['text'],
                                    yaxis2_zerolinecolor=colors[colormode]['text'],
                                    xaxis2_zerolinecolor=colors[colormode]['text'],
                                    yaxis3_gridcolor=colors[colormode]['text'],
                                    xaxis3_gridcolor=colors[colormode]['text'],
                                    yaxis3_zerolinecolor=colors[colormode]['text'],
                                    xaxis3_zerolinecolor=colors[colormode]['text']
                                    )

        return load_plots

    def co2_res(dataset):
        '''
        #2
        'co2_res'
        'XCO2 / SBE O2 / ASVCO2',
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
                                   subplot_titles=(f"Delta xCO{sub2}", f"Mean O{sub2} SBE37", f"Mean O{sub2} ASVCO{sub2}"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=df['time'], y=co2_diff, name='CO2 Diff', hoverinfo='x+y+name',
                                mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=df['time'], y=df['O2_SAT_SBE37_MEAN'], name='O2_SAT_SBE37_MEAN', hoverinfo='x+y+name',
                                mode='markers', marker={'size': mark_size}, row=2, col=1)

        for state in ['EPOFF', 'APOFF']:

            cur_state = df[df['INSTRUMENT_STATE'] == state]
            load_plots.add_scatter(x=cur_state['time'], y=cur_state['O2_MEAN_ASVCO2'], name=state, hoverinfo='x+y+name',
                                    mode='markers', marker={'size': mark_size},  row=3, col=1)


        load_plots['layout'].update(xaxis2_showticklabels=True, xaxis3_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis2_fixedrange=True, yaxis3_fixedrange=True,
                                    yaxis_title=f'CO{sub2} Diff ({mu} mol{inverse} SW-Air)',
                                    yaxis2_title=f'O{sub2} Mean (%)', yaxis3_title=f'O{sub2} Mean (%)',
                                    showlegend=True,
                                    yaxis2_gridcolor=colors[colormode]['text'],
                                    xaxis2_gridcolor=colors[colormode]['text'],
                                    yaxis2_zerolinecolor=colors[colormode]['text'],
                                    xaxis2_zerolinecolor=colors[colormode]['text'],
                                    yaxis3_gridcolor=colors[colormode]['text'],
                                    xaxis3_gridcolor=colors[colormode]['text'],
                                    yaxis3_zerolinecolor=colors[colormode]['text'],
                                    xaxis3_zerolinecolor=colors[colormode]['text']
                                    )

        return load_plots

    def co2_delt(dataset):
        '''
        #3
        'co2_delt'
        'Delta Pressures',
            Primary: calculated pressure differentials between like states
        '''

        df = dataset.get_data(variables=['CO2DETECTOR_PRESS_MEAN_ASVCO2',
                                         'INSTRUMENT_STATE'],
                              window_start=t_start, window_end=t_end)

        df.dropna(axis='rows', subset=['CO2DETECTOR_PRESS_MEAN_ASVCO2'], inplace=True)

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Delta Pressures'],
                                   shared_yaxes=False)

        temp1 = df[df['INSTRUMENT_STATE'] == 'ZPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):

            co2_diff.append(templist1[n] - templist2[n])

        load_plots.add_scatter(x=temp1['time'], y=co2_diff, name='ZP', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'SPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'SPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        load_plots.add_scatter(x=temp1['time'], y=co2_diff, name='SP', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'EPON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'EPOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        load_plots.add_scatter(x=temp1['time'], y=co2_diff, name='EP', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)

        temp1 = df[df['INSTRUMENT_STATE'] == 'APON']
        temp2 = df[df['INSTRUMENT_STATE'] == 'APOFF']

        templist1 = temp1['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()
        templist2 = temp2['CO2DETECTOR_PRESS_MEAN_ASVCO2'].to_list()

        co2_diff = []

        for n in range(len(templist1)):
            co2_diff.append(templist1[n] - templist2[n])

        load_plots.add_scatter(x=temp1['time'], y=co2_diff, name='AP', hoverinfo='x+y+name',
                             mode='markers', marker={'size': mark_size}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='Pressure Differential (kPa)',
                                    showlegend=True)

        return load_plots

    def co2_det_state(dataset):
        '''
        #4
        'co2_det_state'
        'Pump Pressures',
            Primary: CO2DETECTOR_PRESS_MEAN_ASVCO2 for each state
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_PRESS_MEAN_ASVCO2', 'BARO_PRES_MEAN'],
                              window_start=t_start, window_end=t_end)

        temp = df[df['INSTRUMENT_STATE'] == 'ZPON']

        load_plots = make_subplots(rows=2, cols=1,
                                   subplot_titles=['Pump Pressure', 'Barometric Pressure'],
                                   shared_yaxes=False)

        # load_plots.add_scatter(x=temp['time'], y=temp['CO2DETECTOR_PRESS_MEAN_ASVCO2'], name='ZPON', hoverinfo='x+y+name',
        #                        mode='markers', marker={'size': 2}, row=1, col=1)

        for state in states:

             if state == 'SUMMARY':
                continue

             cur_state = df[df['INSTRUMENT_STATE'] == state]

             load_plots.add_scatter(x=cur_state['time'], y=cur_state['CO2DETECTOR_PRESS_MEAN_ASVCO2'],
                                     name=state, hoverinfo='x+y+name',
                                     mode='markers', marker={'size': mark_size},  row=1, col=1)

        load_plots.add_scatter(x=cur_state['time'], y=cur_state['BARO_PRES_MEAN'],
                               name='Baro Mean', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=2, col=1)

        load_plots['layout'].update(yaxis_title='Pump Pressures (kPa)',
                                    yaxis2_title='Barometric Pressure (kPa)',
                                    showlegend=True,
                                    yaxis2_gridcolor=colors[colormode]['text'],
                                    xaxis2_gridcolor=colors[colormode]['text'],
                                    yaxis2_zerolinecolor=colors[colormode]['text'],
                                    xaxis2_zerolinecolor=colors[colormode]['text']
                                    )

        return load_plots

    def co2_mean_zp(dataset):
        '''
        #5
        'co2_mean_zp'
        'Zero Position xCO2 / Detector Temp',
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
                                   subplot_titles=(f"xCO{sub2}", "Temperature"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=primary1['time'], y=primary1['CO2_MEAN_ASVCO2'], name='ZPON', hoverinfo='x+y+name',
                                mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=primary2['time'], y=primary2['CO2_MEAN_ASVCO2'], name='ZPOFF', hoverinfo='x+y+name',
                                mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=primary3['time'], y=primary3['CO2_MEAN_ASVCO2'], name='ZPPCAL', hoverinfo='x+y+name',
                                mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=secondary['time'], y=secondary['CO2DETECTOR_TEMP_MEAN_ASVCO2'].dropna(),
                                name='ZPOFF', hoverinfo='x+y+name',
                                mode='markers', marker={'size': mark_size}, row=2, col=1)

        load_plots['layout'].update(xaxis2_showticklabels=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title=f'CO{sub2} Mean ({mu} mol{inverse})',
                                    yaxis2_title='Temp (°C)',
                                    showlegend=True,
                                    yaxis2_gridcolor=colors[colormode]['text'],
                                    xaxis2_gridcolor=colors[colormode]['text'],
                                    yaxis2_zerolinecolor=colors[colormode]['text'],
                                    xaxis2_zerolinecolor=colors[colormode]['text']
                                    )

        return load_plots

    def co2_mean_sp(dataset):
        '''
        #6
        'co2_mean_sp'
        'Span Position xCO2 / Detector Temp',
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
                                   subplot_titles=(f"Span Position xCO{sub2}", "Detector Temp"),
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=primary1['time'], y=primary1['CO2_MEAN_ASVCO2'], name='SPON', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=primary2['time'], y=primary2['CO2_MEAN_ASVCO2'], name='SPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=primary3['time'], y=primary3['CO2_MEAN_ASVCO2'], name='SPPCAL', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=secondary['time'], y=secondary['CO2DETECTOR_TEMP_MEAN_ASVCO2'].dropna(), name='ZPOFF',
                               hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=2, col=1)


        load_plots['layout'].update(xaxis2_showticklabels=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title=f'xCO{sub2} ({mu} mol{inverse})',
                                    yaxis2_title='Temp (°C)',
                                    showlegend=True,
                                    yaxis2_gridcolor=colors[colormode]['text'],
                                    xaxis2_gridcolor=colors[colormode]['text'],
                                    yaxis2_zerolinecolor=colors[colormode]['text'],
                                    xaxis2_zerolinecolor=colors[colormode]['text']
                                    )

        return load_plots

    def co2_span_temp(dataset):
        '''
        #7
        'co2_span_temp'
        'Span Coefficient vs Temp'
            Primary: CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2 vs. binned SPOFF CO2DETECTOR_TEMP_MEAN_ASVCO2
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_TEMP_MEAN_ASVCO2',
                                         'CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2', 'ASVCO2_ZERO_ERROR_FLAGS'],
                              window_start=t_start, window_end=t_end)

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=['Span Coefficient vs Temp'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        df = df[df['ASVCO2_ZERO_ERROR_FLAGS'] == 0] #drop entries with error codes
        dset = df[df['INSTRUMENT_STATE'] == 'SUMMARY'].sort_values('time', ascending=True).to_numpy()
        df = df[df['INSTRUMENT_STATE'] == 'SPOFF']
        cols = dict(zip(df.columns.to_list(), range(len(df.columns))))

        temps, span, min, max, std, no = [], [], [], [], [], []

        for n, dt in enumerate(dset):

            if n == 0:
                continue

            if dset[n-1, cols['time']] == dt[cols['time']]:
                continue

            group = df[(df['time'] >= dset[n-1, cols['time']]) & (df['time'] < dt[cols['time']])]

            temps.append(group['CO2DETECTOR_TEMP_MEAN_ASVCO2'].mean())
            min.append(group['CO2DETECTOR_TEMP_MEAN_ASVCO2'].min())
            max.append(group['CO2DETECTOR_TEMP_MEAN_ASVCO2'].max())
            std.append(group['CO2DETECTOR_TEMP_MEAN_ASVCO2'].std())
            span.append(dt[cols['CO2DETECTOR_SPAN_COEFFICIENT_ASVCO2']])
            no.append(dset[n-1, cols['time']])

        linear_fit, xm = ls_regression(span, temps)

        bins = pd.DataFrame([temps, std, span, min, max, no]).T.dropna(how='any', axis='rows')
        bins.index = no
        bins.columns = ['temps', 'Temp_STDDEV', 'span', 'Temp_min', 'Temp_max', 'datetime']

        linear_fit = ls_regression(bins['span'], bins['temps'])

        hov_dat = list(zip(min, max, std, no))

        hovertemplate = f'Span Coef: %{{x}}<br>Temp Mean: %{{y}}<br>Temp Min: %{{customdata[0]}}<br>' \
                        f'Temp Max: %{{customdata[1]}} <br>Date: %{{customdata[3]}}'

        load_plots.add_scatter(y=bins['temps'], x=bins['span'],
                               name='Zero Coef vs Temp',
                               customdata=hov_dat, hovertemplate=hovertemplate,#hoverinfo='x+y+name',
                               #error_y=dict(array=bins['Temp STDDEV']),
                               mode='markers', marker={'size': mark_size}, row=1, col=1)

        # load_plots.add_scatter(y=linear_fit, x=xm, name='Least Squares Fit', mode='lines', row=1, col=1)

        load_plots['layout'].update(yaxis_title='Temp (°C)',
                                    xaxis_title='Span Coefficient',
                                    showlegend=False
                                    )

        return load_plots

    def co2_zero_temp(dataset):
        '''
        #8
        'co2_zero_temp'
        'Zero Coeffient vs Temp',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2 vs. binned ZPOFF CO2DETECTOR_TEMP_MEAN_ASVCO2
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_TEMP_MEAN_ASVCO2',
                                         'CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=['Zero Coefficient vs Temp'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        dset = df[df['INSTRUMENT_STATE'] == 'SUMMARY'].sort_values('time', ascending=True).to_numpy()
        df = df[df['INSTRUMENT_STATE'] == 'ZPOFF']
        cols = dict(zip(df.columns.to_list(), range(len(df.columns))))

        temps, span, min, max, std, no = [], [], [], [], [], []


        for n, dt in enumerate(dset):

            if n == 0:
                continue

            if dset[n-1, cols['time']] == dt[cols['time']]:
                continue

            group = df[(df['time'] >= dset[n-1, cols['time']]) & (df['time'] < dt[cols['time']])]

            temps.append(group['CO2DETECTOR_TEMP_MEAN_ASVCO2'].mean())
            min.append(group['CO2DETECTOR_TEMP_MEAN_ASVCO2'].min())
            max.append(group['CO2DETECTOR_TEMP_MEAN_ASVCO2'].max())
            std.append(group['CO2DETECTOR_TEMP_MEAN_ASVCO2'].std())
            span.append(dt[cols['CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2']])
            no.append(dset[n-1, cols['time']])

        bins = pd.DataFrame([temps, std, span, min, max, no]).T
        bins.index = no
        bins.columns = ['temps', 'Temp STDDEV', 'span', 'Temp min', 'Temp max', 'datetime']

        hov_dat = list(zip(min, max, std, no))

        hovertemplate = f'ZERO Coef: %{{x}}<br>Temp Mean: %{{y}}<br>Temp Min: %{{customdata[0]}}<br>' \
                        f'Temp Max: %{{customdata[1]}} <br>Date: %{{customdata[3]}}'

        load_plots.add_scatter(y=bins['temps'], x=bins['span'],
                               name='Zero Coef vs. Temp', hovertemplate=hovertemplate,#hoverinfo='x+y+name',
                               # error_y=dict(array=bins['Temp STDDEV']),
                               customdata=hov_dat,
                               mode='markers', marker={'size': mark_size}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='ZPOFF Detector Temp (°C)',
                                    xaxis_title='Zero Coefficient',
                                    showlegend=False
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

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=[f'CO{sub2} Standard Deviation'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        for n in range(1, len(states)):
            if states[n] == 'SUMMARY':
                continue

            cur_state = df[df['INSTRUMENT_STATE'] == states[n]]
            load_plots.add_scatter(x=cur_state['time'], y=cur_state['CO2_STDDEV_ASVCO2'], name=states[n], hoverinfo='x+y+name',
                                   mode='markers', marker={'size': mark_size}, row=1, col=1)

        load_plots['layout'].update(yaxis_title=f'CO{sub2} Standard Deviation ({mu} mol{inverse})',
                                    showlegend=True)

        return load_plots

    def o2_mean(dataset):
        '''
        #10
        'o2_mean'
        'AVSCO2 O2',
            Primary: O2_MEAN_ASVCO2 for APOFF and EPOFF
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'O2_MEAN_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']
        epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=[f'APOFF and EPOFF O{sub2}'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=apoff['time'], y=apoff['O2_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=epoff['time'], y=epoff['O2_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)

        load_plots['layout'].update(yaxis_title=f'O{sub2} (%)',
                                    )

        return load_plots

    def co2_span(dataset):
        '''
        #11
        'co2_span'
        'Span Coefficient Time Series',
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
                               mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=dset['time'], y=df['CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                               name='Temp Mean', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=2, col=1)

        load_plots['layout'].update(xaxis2_showticklabels=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='Span Coef',
                                    yaxis2_title='Temp (°C)',
                                    showlegend=False,
                                    yaxis2_gridcolor=colors[colormode]['text'],
                                    xaxis2_gridcolor=colors[colormode]['text'],
                                    yaxis2_zerolinecolor=colors[colormode]['text'],
                                    xaxis2_zerolinecolor=colors[colormode]['text']
                                    )

        return load_plots

    def co2_zero(dataset):
        '''
        #12
        'co2_zero'
        'Zero Coefficient vs Time',
            Primary: CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2
            Secondary: CO2DETECTOR_TEMP_MEAN_ASVCO2 for ZPOFF
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2',
                                         'CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                              window_start=t_start, window_end=t_end)

        dset = df[df['INSTRUMENT_STATE'] == 'ZPOFF']

        load_plots = make_subplots(rows=2, cols=1, shared_xaxes='all', subplot_titles=['Zero Coefficient', 'Temperature Mean - ZPOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=df['time'], y=df['CO2DETECTOR_ZERO_COEFFICIENT_ASVCO2'],
                               name='CO2 Zero Coef.', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)
        load_plots.add_scatter(x=dset['time'], y=df['CO2DETECTOR_TEMP_MEAN_ASVCO2'],
                               name='Temp Mean', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=2, col=1)

        load_plots['layout'].update(xaxis2_showticklabels=True,
                                    yaxis2_fixedrange=True,
                                    yaxis_title='Zero Coef',
                                    yaxis2_title='Temp (°C)',
                                    showlegend=False,
                                    yaxis2_gridcolor=colors[colormode]['text'],
                                    xaxis2_gridcolor=colors[colormode]['text'],
                                    yaxis2_zerolinecolor=colors[colormode]['text'],
                                    xaxis2_zerolinecolor=colors[colormode]['text']
                                    )

        return load_plots

    def pres_state(dataset):
        '''
        #13
        'pres_state'
        'AVSCO2 Pressure Check',
            Primary: CO2DETECTOR_PRESS_MEAN_ASVCO2 - BARO_PRES_MEAN &
                    CO2DETECTOR_PRESS_UNCOMP_MEAN_ASVCO2 - BARO_PRES_MEAN
        '''

        df = dataset.get_data(variables=['INSTRUMENT_STATE', 'CO2DETECTOR_PRESS_MEAN_ASVCO2',
                                         'CO2DETECTOR_PRESS_UNCOMP_MEAN_ASVCO2', 'BARO_PRES_MEAN'],
                              window_start=t_start, window_end=t_end)

        uncomp_flag = True

        dset = df[df['INSTRUMENT_STATE'] == 'APOFF']

        cols = dict(zip(df.columns.to_list(), range(len(df.columns))))
        dfnp = df.to_numpy()
        pres_diff = dfnp[:, cols['CO2DETECTOR_PRESS_MEAN_ASVCO2']] - dfnp[:, cols['BARO_PRES_MEAN']]

        try:
            uncomp_diff = dfnp[:, cols['CO2DETECTOR_PRESS_UNCOMP_MEAN_ASVCO2']] - dfnp[:, cols['BARO_PRES_MEAN']]
        except KeyError:
            uncomp_flag = False

        load_plots = make_subplots(rows=1, cols=1, shared_xaxes='all', subplot_titles=['Pressure Differentials - APOFF'],
                                   shared_yaxes=False, vertical_spacing=0.1)

        load_plots.add_scatter(x=dset['time'], y=pres_diff,
                               name='Pres Diff', hoverinfo='x+y+name',
                               mode='markers', marker={'size': mark_size}, row=1, col=1)
        if uncomp_flag:
            load_plots.add_scatter(x=dset['time'], y=uncomp_diff,
                                   name='Uncomp Diff', hoverinfo='x+y+name',
                                   mode='markers', marker={'size': mark_size}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='Presure Difference (kPa)',
                                    showlegend=True)

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

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF', 'SUMMARY']

    dataset = data_import.Dataset(erddap_set)

    plotters = switch_plot(selection)(dataset)

    # bkgrd_colors = {'Dark': 'background',
    #                 'Light': 'light'}

    plotters.update_layout(
         height=600,
         title=' ',
         #hovermode='x unified',
         xaxis_showticklabels=True,
         plot_bgcolor=colors[colormode]['bckgrd'],
         paper_bgcolor=colors[colormode]['bckgrd'],
         font_color=colors[colormode]['text'],
         yaxis_gridcolor=colors[colormode]['text'],
         xaxis_gridcolor=colors[colormode]['text'],
         yaxis_zerolinecolor=colors[colormode]['text'],
         xaxis_zerolinecolor=colors[colormode]['text'],
         autosize=True,
         modebar={'orientation': 'h'},
         margin=dict(l=25, r=25, b=25, t=25, pad=4)
    )

    return plotters

if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)
