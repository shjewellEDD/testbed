'''
Gas Validaiton Dashboard

TODO:
    Plots and filters do not match after changing Display. How to fix this?
        Similar, errors when loading summary tables
    The general callback enforces all filters to singletons instead of lists.
        Is this necessary?
    When the LiCOR is not calibrated it will return -50, we should filter these out by standard
        Look into pump state to find the super high residuals (4000+ values)
            Is this being fixed at the ERDDAP level?
                Yes
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

# urls = [{'label': 'Summary Mirror', 'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/asvco2_gas_validation_summary_mirror.csv'}]

urls = []

with open('validation_sets.csv', 'r') as csv:

    for n, line in enumerate(csv.readlines()):

        if n == 0:
            continue

        urls.append({'label': line.split(',')[0], 'value': line.split(', ')[1]})

raw_urls = pd.read_csv('validation_sets.csv')

custom_sets = [{'label': 'EPOFF & APOFF vs Ref Gas',    'value': 'resids'},
               {'label': 'ZPCAL & SPPCAL vs Ref Gas',   'value': 'cals'},
               #{'label': 'CO2 AVG & STDDEV',            'value': 'temp resids'},
               {'label': 'Residual vs Time',            'value': 'resid stddev'},
               {'label': 'Residual Histogram',          'value': 'stddev hist'},
               {'label': 'Summary Table',               'value': 'summary table'},
               {'label': 'Summary Data',                'value': 'summary data'}]

colors = {'Dark': {'bckgrd': '#111111', 'text': '#7FDBFF'},
          'Light': {'bckgrd': '#FAF9F6', 'text': '#111111'},
          'Green':  '#2ECC40',
          'Blue':   '#0000FF',
          'Red':    '#FF4136'}

mk_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                requests_pathname_prefix='/co2/validation/',
                external_stylesheets=[dbc.themes.SLATE])
server = app.server

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
    '''
    When the datatable is selected, makes the table visible and hides the plot.
    When anything else is selected, hides the table and makes the plot visible.

    :param set_val:
    :return:
    '''

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
    '''
    The master function, entirely dedicated to the plot/table. Controls filtering, display plot, datatable and date
    picker. This feels both messy and necessary given Plotly's restrictions on callbacks.

    Utility functions at the top,
    display plots are controlled by a switch dict at the bottom.
    Each display page has its own function in the middle

    :param plot_set:
    :param plot_fig:
    :param im_mode:
    :param update:
    :param filt1:
    :param filt2:
    :param filt3:
    :param filt4:
    :param filt5:
    :param tstart:
    :param tend:
    :param table_input:
    :return:
    '''

    def gen_filt_list(col_var, dat):
        '''
        Generates dropdown compatiable lists for the filters
        :param uniq_var:
        :param dat:
        :return:
        '''

        filt_list = []

        for var in list(dat[col_var].unique()):

            if not isinstance(var, str):
                continue

            filt_list.append({'label': var, 'value': var})

        return filt_list

    def filter_func(dat, cols, filt_vars):
        '''
        TODO:
        Filters dataframe and concatenates result

        :param dat:
        :param filt_vars:
        :param cols:
        :return:
        '''

        ref_ranges = {'0 thru 750 ppm Range':      ['0ppm', '350ppm Nominal', '500ppm Nominal', '100ppm Nominal'],
                      '0 thru 2 ppm Range':        ['0ppm'],
                      '100 thru 300 ppm Range':    ['100ppm Nominal'],
                      '300 thru 775 ppm Range':    ['350ppm Nominal', '500ppm Nominal', '750ppm Nominal'],
                      '775 thru 1075 ppm Range':   ['1000ppm Nominal'],
                      '1075 thru 2575 ppm Range':  ['1500ppm Nominal', '2000ppm Nominal']
                      }

        # it's possible we'll have these as a pre-existing dict, rather than needing to generate it on the fly
        for col, filt_by in dict(zip(cols, filt_vars)).items():

            temp = []

            for var in filt_by:

                # CO2_DRY_RESIDUAL_REF_LAB_TAG contains ranges (see ref_ranges). These don't actually contain any data,
                # just NaNs. So we'll just contain the relevant nominal ranges. This does give us this ugly if-then
                # block, where without it we could do this entire using pandas commands without a loop
                if var in ref_ranges:
                    temp.append(dat[dat[col].isin(ref_ranges[var])])
                else:
                    temp.append(dat[dat[col] == var])
            dat = pd.merge(dat, pd.concat(temp), how='right')

        return dat

    #an empty table for plot displays
    empty_tables = dcc.Loading([dash_table.DataTable(id='tab1'), dash_table.DataTable(id='tab2')])

    def off_ref(dset):
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
        nonlocal filt1, filt2, filt3, filt4, filt5
        filt_cols = ['OUT_OF_RANGE', 'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'SN_ASVCO2']

        # get dataset
        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_RESIDUAL_STDDEV_ASVCO2',
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

                return dcc.Graph(figure=load_plots), empty_tables, filt_card

            filts = [filt1, filt2, filt3]

            df = filter_func(df, filt_cols, filts)

        # if we are just changing pages, then we need to refresh the filter card
        else:

            filt_list2 = gen_filt_list('CO2_DRY_RESIDUAL_REF_LAB_TAG', df)
            filt_list3 = gen_filt_list('SN_ASVCO2', df)

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

        e_mean = epoff.groupby('CO2_DRY_RESIDUAL_REF_LAB_TAG').mean()
        a_mean = apoff.groupby('CO2_DRY_RESIDUAL_REF_LAB_TAG').mean()
        e_stddev = epoff.groupby('CO2_DRY_RESIDUAL_REF_LAB_TAG').std()
        a_stddev = apoff.groupby('CO2_DRY_RESIDUAL_REF_LAB_TAG').std()

        hovertemplate = f'CO2_REF_LAB: %{{x}}<br>CO2_RESIDUAL_MEAN_ASVCO2: %{{y}}<br>{filt_cols[0]}: %{{customdata[0]}}<br>' \
                        f'{filt_cols[1]}: %{{customdata[1]}} <br> {filt_cols[2]}: %{{customdata[2]}}'

        custom_e = list(zip(epoff[filt_cols[0]], epoff[filt_cols[1]], epoff[filt_cols[2]]))
        custom_a = list(zip(apoff[filt_cols[0]], apoff[filt_cols[1]], apoff[filt_cols[2]]))

        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_DRY_RESIDUAL_MEAN_ASVCO2'],
                               name='EPOFF Dry', customdata=custom_e, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 4, 'color': mk_colors[0]}, row=1, col=1)
        load_plots.add_scatter(x=e_mean['CO2_REF_LAB'], y=e_mean['CO2_DRY_RESIDUAL_MEAN_ASVCO2'],
                               error_y=dict(array=e_stddev['CO2_DRY_RESIDUAL_MEAN_ASVCO2']),
                               name=' Mean EPOFF Dry', mode='markers', marker=dict(size=10, color=mk_colors[0],
                               line=dict(width=2)), row=1, col=1)

        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'],
                               name='EPOFF Dry TCORR', customdata=custom_e, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 4, 'color': mk_colors[1]}, row=1, col=1)
        load_plots.add_scatter(x=e_mean['CO2_REF_LAB'], y=e_mean['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'],
                               error_y=dict(array=e_stddev['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2']),
                               name='Mean EPOFF TCORR', mode='markers', marker=dict(size=10, color=mk_colors[1],
                               line=dict(width=2)), row=1, col=1)

        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_DRY_RESIDUAL_MEAN_ASVCO2'],
                               name='APOFF Dry', customdata=custom_a, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 4, 'color': mk_colors[2]}, row=1, col=1)
        load_plots.add_scatter(x=a_mean['CO2_REF_LAB'], y=a_mean['CO2_DRY_RESIDUAL_MEAN_ASVCO2'],
                               error_y=dict(array=a_stddev['CO2_DRY_RESIDUAL_MEAN_ASVCO2']),
                               name='Mean APOFF Dry', mode='markers', marker=dict(size=10, color=mk_colors[2],
                               line=dict(width=2)), row=1, col=1)

        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'],
                               name='APOFF Dry TCORR', customdata=custom_a, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 4, 'color': mk_colors[3]}, row=1, col=1)
        load_plots.add_scatter(x=a_mean['CO2_REF_LAB'], y=a_mean['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'],
                               error_y=dict(array=a_stddev['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2']),
                               name='Mean APOFF TCORR', mode='markers', marker=dict(size=10, color=mk_colors[3],
                               line=dict(width=2)), row=1, col=1)

        load_plots['layout'].update(yaxis_title='Residual (ppm)',
                                    xaxis_title='Reference Gas Concentration (ppm)'
                                    )

        return dcc.Graph(figure=load_plots), empty_tables, filt_card


    def cal_ref(dset):
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
        nonlocal filt1, filt2, filt3, filt4, filt5
        filt_cols = ['OUT_OF_RANGE', 'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'SN_ASVCO2']
        filts = [filt1, filt2, filt3]

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

                return dcc.Graph(figure=load_plots), empty_tables, filt_card

            df = filter_func(df, filt_cols, filts)

        # if we are just changing pages, then we need to refresh the filter card
        else:

            filt_list2 = gen_filt_list('CO2_DRY_RESIDUAL_REF_LAB_TAG', df)
            filt_list3 = gen_filt_list('SN_ASVCO2', df)

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

        hovertemplate = f'CO2 Reference: %{{x}}<br>Residual: %{{y}}<br>{filt_cols[0]}: %{{customdata[0]}}<br>' \
                        f'{filt_cols[1]}: %{{customdata[1]}} <br> {filt_cols[2]}: %{{customdata[2]}}'

        zcal = df[df['INSTRUMENT_STATE'] == 'ZPPCAL']
        scal = df[df['INSTRUMENT_STATE'] == 'SPPCAL']

        custom_z = list(zip(zcal[filt_cols[0]], zcal[filt_cols[1]], zcal[filt_cols[2]]))
        custom_s = list(zip(scal[filt_cols[0]], scal[filt_cols[1]], scal[filt_cols[2]]))

        z_mean = zcal.groupby('CO2_DRY_RESIDUAL_REF_LAB_TAG').mean()
        s_mean = scal.groupby('CO2_DRY_RESIDUAL_REF_LAB_TAG').mean()
        z_stddev = zcal.groupby('CO2_DRY_RESIDUAL_REF_LAB_TAG').std()
        s_stddev = scal.groupby('CO2_DRY_RESIDUAL_REF_LAB_TAG').std()

        load_plots.add_scatter(x=zcal['CO2_REF_LAB'], y=zcal['CO2_RESIDUAL_MEAN_ASVCO2'],
                               name='ZPPCAL', customdata=custom_z, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 4, 'color': mk_colors[0]}, row=1, col=1)
        load_plots.add_scatter(x=z_mean['CO2_REF_LAB'], y=z_mean['CO2_RESIDUAL_MEAN_ASVCO2'],
                               error_y=dict(array=z_stddev['CO2_RESIDUAL_MEAN_ASVCO2']),
                               name='Mean ZPPCAL', mode='markers', marker=dict(size=10, color=mk_colors[0],
                               line=dict(width=2)), row=1, col=1)

        load_plots.add_scatter(x=scal['CO2_REF_LAB'], y=scal['CO2_RESIDUAL_MEAN_ASVCO2'],
                               name='SPPCAL', customdata=custom_s, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 4, 'color': mk_colors[1]}, row=1, col=1)
        load_plots.add_scatter(x=s_mean['CO2_REF_LAB'], y=s_mean['CO2_RESIDUAL_MEAN_ASVCO2'],
                               error_y=dict(array=s_stddev['CO2_RESIDUAL_MEAN_ASVCO2']),
                               name='Mean SPPCAL', mode='markers', marker=dict(size=10, color=mk_colors[1],
                               line=dict(width=2)), row=1, col=1)

        load_plots['layout'].update(
                                    yaxis_title='Residual (ppm)',
                                    xaxis_title='Reference Gas (ppm)',
                                    )

        return dcc.Graph(figure=load_plots), empty_tables, filt_card

    def multi_ref(dset):
        '''
        Currently disabled; it's been integrated into the ZPCAL/SPCAL dashboard

        "CO2 AVG & STDDEV"
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Residual
        :return:
        TODO:

        '''
        nonlocal filt1, filt2, filt3, filt4, filt5

        filt_cols = ['SN_ASVCO2', 'CO2DETECTOR_firmware', 'ASVCO2_firmware', 'last_ASVCO2_validation']
        filts = [filt1, filt2, filt3, filt4]

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

                return dcc.Graph(figure=load_plots), empty_tables, filt_card

            df = filter_func(df, filt_cols, filts)

        else:

            filt_list1 = gen_filt_list('SN_ASVCO2', df)
            filt_list3 = gen_filt_list('ASVCO2_firmware', df)
            filt_list4 = gen_filt_list('last_ASVCO2_validation', df)

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

        customdata = list(zip(df[filt_cols[0]], df[filt_cols[1]], df[filt_cols[2]], df[filt_cols[3]]))

        hovertemplate = f'CO2 Reference: %{{x}}<br>Residual: %{{y}} <br> {filt_cols[0]}: %{{customdata[0]}}<br>' \
                        f'{filt_cols[1]}: %{{customdata[1]}} <br> {filt_cols[2]}: %{{customdata[2]}}<br>' \
                        f'{filt_cols[3]}: %{{customdata[3]}}'

        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_RESIDUAL_MEAN_ASVCO2'], name='Residual',
                               customdata=customdata, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_DRY_RESIDUAL_MEAN_ASVCO2'], name='Dry Resid',
                               customdata=customdata, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='TCORR Dry Resid',
                               customdata=customdata, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=df['CO2_REF_LAB'], y=df['CO2_RESIDUAL_STDDEV_ASVCO2'], name='Residual STDDEV',
                               customdata=customdata, hovertemplate=hovertemplate,
                               mode='markers', marker={'size': 5}, row=2, col=1)

        load_plots['layout'].update(yaxis1_title='CO2 Residuals (ppm)',
                                    xaxis1_title='CO2 Gas Concentration (ppm)',
                                    yaxis2_title='CO2 STDDEV (ppm)',
                                    xaxis2_title='CO2 Gas Concentration (ppm)',
                                    )


        return dcc.Graph(figure=load_plots), empty_tables, filt_card


    def resid_and_stdev(dset):
        '''
        "Resid vs Time"
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Residual vs time, with STDDEV as error bars
        :return:

        TODO:
            CO2_RESIDUAL_STDDEV_ASVCO2 is all 0. This isn't correct, is it? It is!
            So, should STDDEV be from CO2_DRY_RESIDUAL_MEAN_ASVCO2

        NOTES:
            At test, the Last Validation filter doesn't appear to work.
        '''
        
        nonlocal filt1, filt2, filt3, filt4, filt5

        filt_cols = ['SN_ASVCO2', 'CO2DETECTOR_firmware', 'ASVCO2_firmware', 'last_ASVCO2_validation',
                     'CO2_DRY_RESIDUAL_REF_LAB_TAG']
        filts = [filt1, filt2, filt3, filt4, filt5]

        df = dset.get_data(variables=['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'INSTRUMENT_STATE',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'CO2_RESIDUAL_STDDEV_ASVCO2', 'CO2_REF_LAB',
                                      'SN_ASVCO2', 'ASVCO2_firmware', 'CO2DETECTOR_firmware', 'last_ASVCO2_validation',
                                      'CO2_DRY_RESIDUAL_REF_LAB_TAG'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Residual Over Time'],
                                   shared_yaxes=False)

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == [] or filt4 == [] or filt5 == []:
                load_plots = make_subplots(rows=1, cols=1,
                                           shared_yaxes=False, shared_xaxes=True)

                return dcc.Graph(figure=load_plots), empty_tables,  filt_card

            df = filter_func(df, filt_cols, filts)

        else:

            filt_list1 = gen_filt_list('SN_ASVCO2', df)
            filt_list3 = gen_filt_list('ASVCO2_firmware', df)
            filt_list4 = gen_filt_list('last_ASVCO2_validation', df)

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
                         dhtml.Label('CO2 Reference Tag'),
                         dcc.Dropdown(id='filter5', options=df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique(),
                                      multi=True, clearable=True, value=df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique()),
                         dhtml.Button('Update Filter', id='update')]

        # plotting block
        customdata = list(zip(df[filt_cols[0]], df[filt_cols[1]], df[filt_cols[2]], df[filt_cols[3]], df[filt_cols[4]]))

        hovertemplate = f'CO2 Reference: %{{x}}<br>Residual: %{{y}} <br> {filt_cols[0]}: %{{customdata[0]}}<br>' \
                        f'{filt_cols[1]}: %{{customdata[1]}} <br> {filt_cols[2]}: %{{customdata[2]}}<br>' \
                        f'{filt_cols[3]}: %{{customdata[3]}} <br> {filt_cols[4]}: %{{customdata[4]}}'

        for n, inst_state in enumerate(['APOFF', 'EPOFF']):

            temp = df[df['INSTRUMENT_STATE'] == inst_state]

            dry_mean = temp.groupby('last_ASVCO2_validation')['CO2_DRY_RESIDUAL_MEAN_ASVCO2'].mean()
            dry_stddev = temp.groupby('last_ASVCO2_validation')['CO2_DRY_RESIDUAL_MEAN_ASVCO2'].std()
            tcorr_mean = temp.groupby('last_ASVCO2_validation')['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'].mean()
            tcorr_stddev = temp.groupby('last_ASVCO2_validation')['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'].std()
            time_mean = temp.groupby('last_ASVCO2_validation')['time'].mean()

            # load_plots.add_scatter(x=temp['time'], y=temp['CO2_RESIDUAL_MEAN_ASVCO2'],
            #                        error_y=dict(array=temp['CO2_RESIDUAL_STDDEV_ASVCO2']),
            #                        name=f'{inst_state} Residual', customdata=customdata, hovertemplate=hovertemplate,
            #                        mode='markers', marker={'size': 4}, row=1, col=1)
            load_plots.add_scatter(x=temp['time'], y=temp['CO2_DRY_RESIDUAL_MEAN_ASVCO2'],
                                   name=f'{inst_state} Dry Residual', customdata=customdata, hovertemplate=hovertemplate,
                                   mode='markers', marker=dict(size=4, color=mk_colors[n*2]), row=1, col=1, )
            load_plots.add_scatter(x=temp['time'], y=temp['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'],
                                   customdata=customdata, hovertemplate=hovertemplate,
                                   name=f'{inst_state} Dry TCORR Residual', mode='markers', row=1, col=1,
                                   marker=dict(size= 4, color=mk_colors[(n*2)+1]))

            load_plots.add_scatter(x=time_mean, y=dry_mean.to_list(),
                                   error_y=dict(array=dry_stddev.to_list()),
                                   mode='markers', marker=dict(size=10, color=mk_colors[n*2], line=dict(width=2)), row=1, col=1,
                                   name=f'{inst_state} Dry Mean')
            load_plots.add_scatter(x=time_mean, y=tcorr_mean.to_list(),
                                   error_y=dict(array=tcorr_stddev.to_list()),
                                   mode = 'markers', marker=dict(size=10, color=mk_colors[(n*2)+1],
                                                                line=dict(width=2)), row = 1, col = 1,
                                   name=f'{inst_state} TCORR Mean')

        load_plots['layout'].update(
            yaxis_title='Residual w/ Standard Deviation (ppm)'
        )

        return dcc.Graph(figure=load_plots), empty_tables, filt_card

    def stddev_hist(dset):
        '''
        "Residual Histogram"
        Select random variable
        Histogram of marginal probability dists

        :return:
        '''
        nonlocal filt1, filt2, filt3, filt4, filt5

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

                return dcc.Graph(figure=load_plots), empty_tables, filt_card

            # ASVCO2_firmware is a special case
            filt_cols = ['ASVCO2_firmware', 'INSTRUMENT_STATE', 'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'last_ASVCO2_validation', 'CO2DETECTOR_firmware']
            filts = [filt1, filt2, filt3, filt4, filt5]

            df = filter_func(df, filt_cols, filts)

        else:

            filt_list1 = gen_filt_list('ASVCO2_firmware', df)
            filt_list3 = gen_filt_list('CO2_DRY_RESIDUAL_REF_LAB_TAG', df)
            filt_list4 = gen_filt_list('last_ASVCO2_validation', df)

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

    def summary_table(dset):
        '''
        TODO:

        The filters here seem to work, despite being a nightmare.

        Returns

        :return:
        '''
        nonlocal filt1, filt2, filt3, table_input

        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2', 'SN_ASVCO2',
                                 'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'ASVCO2_firmware',
                                 'CO2_RESIDUAL_STDDEV_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'OUT_OF_RANGE'])

        df = df[(df['INSTRUMENT_STATE'] == 'APOFF') | (df['INSTRUMENT_STATE'] == 'EPOFF')]

        limiters = {'pf_mean':       1,
                    'pf_stddev':     .5,
                    'pf_max':        2
                    }

        default = [{'sn': '', 'mean': 'Mean Residual', 'stddev': 'STDDEV', 'max': 'Max Residual'},
                   {'sn': 'Pass/Fail', 'mean': 1, 'stddev': .5, 'max': 2},  # defaults
                   {'sn': 'APOFF Fail %', 'mean': '', 'stddev': '', 'max': ''},
                   {'sn': 'EPOFF Fail %', 'mean': '', 'stddev': '', 'max': ''}
                   ]

        # default values
        if (filt1 == []) or (filt1 == [None]):
            filt1 = 'CO2_DRY_RESIDUAL_MEAN_ASVCO2'
        else:
            filt1 = filt1[0]

        if (filt2 == []) or (filt2 == [None]) or (set(filt2) == set(['APOFF', 'EPOFF'])):
            filt2 = 'Both'
        else:
            filt2 = filt2[0]

        filt3 = filt3[0]

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # double check this
            if filt3 != None:
                df = df[df['CO2_DRY_RESIDUAL_REF_LAB_TAG'] == filt3]

            # supposedly dash_tables will have built-in typing at some point in the future... this might make them even
            # more of a mess on the backend, but will help eliminate this mess of a guard clause

            if (table_input[1]['mean'] is not None) or (table_input[1]['mean'] != ''):
                try:
                    limiters['pf_mean'] = float(table_input[1]['mean'])
                    default[1]['mean'] = float(table_input[1]['mean'])
                except (ValueError, TypeError):
                    limiters['pf_mean'] = ''
                    default[1]['mean'] = ''

            if (table_input[1]['stddev'] is not None) or (table_input[1]['stddev'] != ''):
                try:
                    limiters['pf_stddev'] = float(table_input[1]['stddev'])
                    default[1]['stddev'] = float(table_input[1]['stddev'])
                except (ValueError, TypeError):
                    limiters['pf_stddev'] = ''
                    default[1]['stddev'] = ''

            if (table_input[1]['max'] is not None) or (table_input[1]['max'] != ''):
                try:
                    limiters['pf_max'] = float(table_input[1]['max'])
                    default[1]['max'] = float(table_input[1]['max'])
                except (ValueError, TypeError):
                    limiters['pf_max'] = ''
                    default[1]['max'] = ''

        else:

            filt_list1 = [{'label': 'Dry Residual',    'value': 'CO2_DRY_RESIDUAL_MEAN_ASVCO2'},
                          {'label': 'TCORR Residual',  'value': 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'},
                          {'label': 'STDDEV',          'value': 'CO2_RESIDUAL_STDDEV_ASVCO2'}
                          ]

            filt_list3 = []


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

        for sn in df['SN_ASVCO2'].unique():

            # this seems like a silly way to check for existance. Maybe change
            if sn in df['SN_ASVCO2'].unique():
                temp['Both'][sn] = {'sn': sn,
                            'mean': float(df[df['SN_ASVCO2'] == sn][filt1].mean()),
                            'stddev': float(df[df['SN_ASVCO2'] == sn][filt1].std()),
                            'max': float(df[df['SN_ASVCO2'] == sn][filt1].max())}

            else:
                temp['Both'][sn] = {'sn': sn,
                            'mean': '',
                            'stddev': '',
                            'max': ''}

            for state in ['APOFF', 'EPOFF']:

                if sn in df['SN_ASVCO2'].unique():
                    temp[state][sn] = {'sn': sn,
                                        'mean': float(df[(df['SN_ASVCO2'] == sn) &
                                                   (df['INSTRUMENT_STATE'] == state)][filt1].mean()),
                                        'stddev': float(df[(df['SN_ASVCO2'] == sn) &
                                                     (df['INSTRUMENT_STATE'] == state)][filt1].std()),
                                        'max': float(df[(df['SN_ASVCO2'] == sn) &
                                                  (df['INSTRUMENT_STATE'] == state)][filt1].max())}

                else:
                    temp[state][sn] = {'sn': sn,
                                        'mean': '',
                                        'stddev': '',
                                        'max': ''}

        table_data = {'Both':   pd.DataFrame.from_dict(temp['Both'], orient='index'),
                      'APOFF':  pd.DataFrame.from_dict(temp['APOFF'], orient='index'),
                      'EPOFF':  pd.DataFrame.from_dict(temp['EPOFF'], orient='index')}
        temp = []

        if table_data[filt2]['mean'].dropna().empty:
            default[2]['mean'], default[3]['mean'] = '', ''

        elif limiters['pf_mean']:

            temp = table_data['APOFF']['mean'].dropna()
            perc = 100 * len(temp[abs(temp) > limiters["pf_mean"]]) / len(temp)

            default[2]['mean'] = f'{perc}%'

            temp = table_data['EPOFF']['mean'].dropna()
            perc = 100 * len(temp[abs(temp) > limiters["pf_mean"]]) / len(temp)

            default[3]['mean'] = f'{perc}%'

        # STDDEV pass/fail percentage
        if table_data[filt2]['stddev'].dropna().empty:
            default[2]['stddev'], default[3]['stddev'] = '', ''

        elif limiters['pf_stddev']:

            temp = table_data['APOFF']['stddev'].dropna()
            perc = 100 * len(temp[abs(temp) > limiters["pf_stddev"]]) / len(temp)
            default[2]['stddev'] = f'{perc}%'

            temp = table_data['EPOFF']['stddev'].dropna()
            perc = 100 * len(temp[abs(temp) > limiters["pf_stddev"]]) / len(temp)
            default[3]['stddev'] = f'{perc}%'

        # Max pass/fail percentage
        if table_data[filt2]['max'].dropna().empty:
            default[2]['max'], default[3]['max'] = '', ''

        elif limiters['pf_max']:

            temp = table_data['APOFF']['max'].dropna()
            perc = 100 * len(temp[abs(temp) > limiters["pf_max"]]) / len(temp)
            default[2]['max'] = f'{perc}%'

            temp = table_data['EPOFF']['max'].dropna()
            perc = 100 * len(temp[abs(temp) > limiters["pf_max"]]) / len(temp)
            default[3]['max'] = f'{perc}%'

        def_df = pd.DataFrame.from_dict(default)
        def_df.fillna('')

        tab1 = dash_table.DataTable(def_df.to_dict('records'),
                                    columns=[{"name": i, "id": i, 'editable': True} for i in def_df.columns],
                                    id='tab1',
                                    style_table={'backgroundColor': colors[im_mode]['bckgrd']},
                                    style_cell={'backgroundColor': colors[im_mode]['bckgrd'],
                                                'textColor': colors[im_mode]['text']},
                                    style_data_conditional=[
                                        {
                                            'if': {
                                                'row_index': 1,
                                            },
                                            'fontWeight': 'bold',
                                            'color':      'green'
                                        },
                                        {
                                            'if': {
                                                'column_id':    'mean',
                                                'filter_query': '("APOFF Fail %" = {sn} or "EPOFF Fail %" = {sn}) && {mean} = "0.0%"',
                                            },
                                            'backgroundColor': colors['Green'],
                                            'color':       'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'stddev',
                                                'filter_query': '("APOFF Fail %" = {sn} or "EPOFF Fail %" = {sn}) && {stddev} = "0.0%"',
                                            },
                                            'backgroundColor': colors['Green'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'max',
                                                'filter_query': '("APOFF Fail %" = {sn} or "EPOFF Fail %" = {sn}) && {max} = "0.0%"',
                                            },
                                            'backgroundColor': colors['Green'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'mean',
                                                'filter_query': '("APOFF Fail %" = {sn} or "EPOFF Fail %" = {sn}) && {mean} != "0.0%"'
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'stddev',
                                                'filter_query': '("APOFF Fail %" = {sn} or "EPOFF Fail %" = {sn}) && {stddev} != "0.0%"'
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'max',
                                                'filter_query': '("APOFF Fail %" = {sn} or "EPOFF Fail %" = {sn}) && {max} != "0.0%"'
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        }
                                    ]
                                    )

        tab2 = dash_table.DataTable(table_data[filt2].to_dict('records'),
                                    columns=[{"name": i, "id": i} for i in table_data[filt2].columns],
                                    id='tab2',
                                    sort_action='native',
                                    style_table={'backgroundColor': colors[im_mode]['bckgrd']},
                                    style_cell={'backgroundColor': colors[im_mode]['bckgrd'],
                                                'textColor': colors[im_mode]['text']},
                                    style_data_conditional=[
                                        {
                                            'if': {
                                                'column_id': 'mean',
                                                'filter_query': '{{mean}} > {}'.format(limiters['pf_mean']),
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'mean',
                                                'filter_query': '{{mean}} < {}'.format(-1*limiters['pf_mean']),
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'stddev',
                                                'filter_query': '{{stddev}} > {}'.format(limiters['pf_stddev']),
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'stddev',
                                                'filter_query': '{{stddev}} < {}'.format(-1*limiters['pf_stddev']),
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'max',
                                                'filter_query': '{{max}} > {}'.format(limiters['pf_max']),
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        },
                                        {
                                            'if': {
                                                'column_id': 'max',
                                                'filter_query': '{{max}} < {}'.format(-1 * limiters['pf_max']),
                                            },
                                            'backgroundColor': colors['Red'],
                                            'color': 'black'
                                        },
                                    ]
                                    )

        return dcc.Graph(id='graphs'), [dcc.Loading(tab1), dcc.Loading(tab2)], filt_card

    def summary(dset):
        '''
        TODO:

        Returns

        :return:
        '''

        nonlocal filt1, filt2, filt3, filt4, filt5

        filt_cols = ['SN_ASVCO2', 'ASVCO2_firmware', 'last_ASVCO2_validation', 'CO2DETECTOR_firmware',
                     'CO2_DRY_RESIDUAL_REF_LAB_TAG']
        filts = [filt1, filt2, filt3, filt4, filt5]

        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2', 'OUT_OF_RANGE',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_RESIDUAL_REF_LAB_TAG', 'SN_ASVCO2', 'ASVCO2_firmware',
                                      'CO2DETECTOR_firmware', 'last_ASVCO2_validation', 'CO2_RESIDUAL_STDDEV_ASVCO2'])

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Cal vs Reference'],
                                   shared_yaxes=False)

        # filter block
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'update.n_clicks' in changed_id:

            filt_card = dash.no_update

            # if we're filtering everything, don't worry about plotting
            if filt1 == [] or filt2 == [] or filt3 == [] or filt4 == [] or filt5 == []:

                return dcc.Graph(figure=load_plots), empty_tables, filt_card

            df = filter_func(df, filt_cols, filts)

        # if we are just changing pages, then we need to refresh the filter card
        else:

            filt_list1 = gen_filt_list('SN_ASVCO2', df)
            filt_list2 = gen_filt_list('ASVCO2_firmware', df)
            filt_list3 = gen_filt_list('last_ASVCO2_validation', df)
            filt_list4 = gen_filt_list('CO2DETECTOR_firmware', df)
            filt_list5 = gen_filt_list('CO2_DRY_RESIDUAL_REF_LAB_TAG', df)

            filt_card = [dcc.DatePickerRange(id='date-picker'),
                         dhtml.Label('Serial #'),
                         dcc.Dropdown(id='filter1', options=filt_list1, value=df['SN_ASVCO2'].unique(), multi=True,
                                      clearable=True),
                         dhtml.Label('Firmware'),
                         dcc.Dropdown(id='filter2', options=filt_list2, value=df['ASVCO2_firmware'].unique(), multi=True,
                                      clearable=True),
                         dhtml.Label(id='Last Validation'),
                         dcc.Dropdown(id='filter3', options=filt_list3, value=df['last_ASVCO2_validation'].unique(),
                                      multi=True, clearable=True),
                         dhtml.Label(id='LiCOR Firmware'),
                         dcc.Dropdown(id='filter4', options=filt_list4, value=df['CO2DETECTOR_firmware'].unique(),
                                      multi=True, clearable=True),
                         dhtml.Label(id='Lab Reference CO2'),
                         dcc.Dropdown(id='filter5', options=filt_list5, value=df['CO2_DRY_RESIDUAL_REF_LAB_TAG'].unique(),
                                      multi=True, clearable=True),
                         dhtml.Button('Update Filter', id='update')]

        data_mean = df.groupby('last_ASVCO2_validation').mean()
        data_stddev = df.groupby('last_ASVCO2_validation').std()

        # data_mean = df.groupby('CO2_REF_LAB').mean()
        # data_stddev = df.groupby('CO2_REF_LAB').std()

        customdata = list(zip(df[filt_cols[0]], df[filt_cols[1]], df[filt_cols[2]], df[filt_cols[3]], df[filt_cols[4]]))

        hovertemplate = f'CO2 Reference: %{{x}}<br>Residual: %{{y}} <br> {filt_cols[0]}: %{{customdata[0]}}<br>' \
                        f'{filt_cols[1]}: %{{customdata[1]}} <br> {filt_cols[2]}: %{{customdata[2]}}' \
                        f'<br> {filt_cols[3]}: %{{customdata[3]}} <br> {filt_cols[4]}: %{{customdata[4]}}'

        for resid_type in ['CO2_DRY_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2']:

            load_plots.add_scatter(x=df['CO2_REF_LAB'].unique(), y=data_mean[resid_type], name=resid_type,
                                   error_y=dict(array=data_stddev[resid_type]),
                                   customdata=customdata, hovertemplate=hovertemplate,
                                   mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(
                                    yaxis_title='Residual (ppm)',
                                    xaxis_title='Reference Gas (ppm)',
                                    )

        return dcc.Graph(figure=load_plots), empty_tables, filt_card

    # enforce filter return as list
    if not isinstance(filt1, list):
        filt1 = [filt1]
    if not isinstance(filt2, list):
        filt2 = [filt2]
    if not isinstance(filt3, list):
        filt3 = [filt3]
    if not isinstance(filt4, list):
        filt4 = [filt4]
    if not isinstance(filt5, list):
        filt5 = [filt5]

    def switch_plot(case):
        return {'resids':        off_ref,
                'cals':          cal_ref,
                'temp resids':   multi_ref,
                'resid stddev':  resid_and_stdev,
                'stddev hist':   stddev_hist,
                'summary table': summary_table,
                'summary data':  summary
                }.get(case)

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

    dataset = data_import.Dataset(plot_set, window_start=tstart, window_end=tend)
    plotters = switch_plot(plot_fig)(dataset)

    if plot_fig == 'summary table':
        pass

    else:
        plotters[0].figure.update_layout(height=600,
            title=' ',
            xaxis_showticklabels=True,
            yaxis_fixedrange=False,
            plot_bgcolor=colors[im_mode]['bckgrd'],
            paper_bgcolor=colors[im_mode]['bckgrd'],
            font_color=colors[im_mode]['text'],
            yaxis_gridcolor=colors[im_mode]['text'],
            xaxis_gridcolor=colors[im_mode]['text'],
            yaxis_zerolinecolor=colors[im_mode]['text'],
            xaxis_zerolinecolor=colors[im_mode]['text'],
            #xaxis=dict(showgrid=False),
            showlegend=True, modebar={'orientation': 'h'},
            margin=dict(l=25, r=25, b=25, t=25, pad=4)
        )

    return plotters[0], plotters[1], plotters[2]


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)
