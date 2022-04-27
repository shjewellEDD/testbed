'''
Gas Validaiton Dashboard

TODO:
    What column defines if the point has failed or passed the range check?
    Need to come up with a way to deal with sets and make it come from outside the data_import.py module
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

state_select_vars = ['INSTRUMENT_STATE', 'last_ASVCO2_validation', 'CO2LastZero', 'ASVCO2_firmware',
                     'CO2DETECTOR_serialnumber', 'ASVCO2_ATRH_serialnumber', 'ASVCO2_O2_serialnumber',
                     'ASVCO2_vendor_name']

resid_vars = ['CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_RESIDUAL_STDDEV_ASVCO2', ' CO2_DRY_RESIDUAL_MEAN_ASVCO2', ' CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2',
              ]

#set_url = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/asvco2_gas_validation_summary_mirror.csv'

urls = [{'label': 'Summary Mirror', 'value': 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/asvco2_gas_validation_summary_mirror.csv'}]

custom_sets = [{'label': 'XCO2 Mean',       'value': 'resids'},
               {'label': 'XCO2 Residuals',  'value': 'cals'},
               {'label': 'CO2 STDDEV',      'value': 'temp resids'},
               {'label': 'CO2 Pres. Mean',  'value': 'stddev'},
               {'label': 'CO2 Mean',        'value': 'resid stddev'}]

#dataset = data_import.Dataset(urls[0]['value'])

colors = {'Dark': '#111111', 'Light': '#443633', 'text': '#7FDBFF'}

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                #requests_pathname_prefix='/co2/validation/',
                external_stylesheets=[dbc.themes.SLATE])
#server = app.server

tools_card = dbc.Card([
    dbc.CardBody(
           style={'backgroundColor': colors['Dark']},
           children=[
                # dcc.DatePickerRange(
                # id='date-picker',
                # min_date_allowed=dataset.t_start,
                # max_date_allowed=dataset.t_end,
                # start_date=dataset.t_end - datetime.timedelta(days=14),
                # end_date=dataset.t_end
                # ),
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
                ),
            dash_table.DataTable(
                id='datatable',
                #data=dataset.to_dict('records'),
                # columns=[{'name': 'Serial Number', 'id': 'serial'},
                #          {'name': 'Size', 'id': 'size'},
                #          {'name': 'State', 'id': 'state'}]
                )
    ])
])

graph_card = dbc.Card(
    [dbc.CardBody(
         [dcc.Loading(dcc.Graph(id='graphs'))])
    ]
)


app.layout = dhtml.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([dhtml.H1('ASVCO2 Validation Set')]),
            dbc.Row([
                dbc.Col(children=[tools_card,
                                  dcc.RadioItems(id='image_mode',
                                                 options=['Dark', 'Light'],
                                                 value='Dark')
                                  ],
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

# plot updating selection
@app.callback(
    [Output('graphs', 'figure')],
     #Output('datatable', 'data'),
     #Output('datatable', 'columns')],
    [Input('select_set', 'value'),
     Input('select_display', 'value'),
     Input('image_mode', 'value')
     # Input('date-picker', 'start_date'),
     # Input('date-picker', 'end_date')
     ])

def load_plot(plot_set, plot_fig, im_mode):

    def off_ref(dset):
        '''
        TODO:
            Hoverdata should be richer
            Better selection of colors
            Change colors by OUT_OF_RANGE flag

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

        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2',
                                      'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'])

        epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']
        apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Pressure'],
                                   shared_yaxes=False, shared_xaxes=True)

        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_RESIDUAL_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_RESIDUAL_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(yaxis_title='Residual',
                                    xaxis_title='CO2 Gas Concentration'
                                    )

        # dtable = dash_table.DataTable()
        #
        # columns = [{'id': 'state', 'name': 'State'},
        #          {'id': 'size', 'name': 'Size'}]
        #
        # table_df = pd.concat([epoff, apoff])
        # drivers = [list(table_df['INSTRUMENT_STATE'].unique()), list(table_df.groupby('INSTRUMENT_STATE').size())]
        # sizes = [str(x[0]) + ", " + str(x[1]) for x in drivers]
        #
        # table_data = [{'state': list(table_df['OUT_OF_RANGE'].unique())},
        #               {'size': sizes}]
        #
        # return load_plots, table_data, columns
        return load_plots

    def cal_ref(dset):
        '''
        Select serial and date
            serial: SN_ASVCO2
            date: time
        Residual for ZPPCAL and SPPCAL vs gas concentration
            state: INSTRUMENT_STATE
            gas:
        :return:
        '''

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

        # #dtable = dash_table.DataTable()
        #
        # columns = [{'id': 'state', 'name': 'State'},
        #          {'id': 'size', 'name': 'Size'}]
        #
        # table_df = pd.concat([epoff, apoff])
        # drivers = [list(table_df['INSTRUMENT_STATE'].unique()), list(table_df.groupby('INSTRUMENT_STATE').size())]
        # sizes = [str(x[0]) + ", " + str(x[1]) for x in drivers]
        #
        # table_data = [{'state': list(table_df['INSTRUMENT_STATE'].unique())},
        #                      {'size': sizes}]

        return load_plots  # , table_data, columns

        #return


    def multi_ref(dset):
        '''
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Residual
        :return:
        TODO:
            Add INSTRUMENT_STATE to hoverinfo
        '''

        df = dset.get_data(variables=['INSTRUMENT_STATE', 'CO2_REF_LAB', 'CO2_RESIDUAL_MEAN_ASVCO2', 'CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2',
                            'CO2_RESIDUAL_STDDEV_ASVCO2', 'CO2_STDDEV_ASVCO2'])

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

        # #dtable = dash_table.DataTable()
        #
        # columns = [{'id': 'state', 'name': 'State'},
        #          {'id': 'size', 'name': 'Size'}]
        #
        # table_df = pd.concat([epoff, apoff])
        # drivers = [list(table_df['INSTRUMENT_STATE'].unique()), list(table_df.groupby('INSTRUMENT_STATE').size())]
        # sizes = [str(x[0]) + ", " + str(x[1]) for x in drivers]
        #
        # table_data = [{'state': list(table_df['INSTRUMENT_STATE'].unique())},
        #                      {'size': sizes}]

        return load_plots  # , table_data, columns


    def multi_stddev(df):
        '''
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Standard deviation
        :return:
        '''

        return


    def resid_and_stdev(df):
        '''
        Select random variable
        Histogram of marginal probability dists
        :return:
        '''

        return

    def switch_plot(case):
        return {'resids':       off_ref,
                'cals':         cal_ref,
                'temp resids':  multi_ref,
                'stddev':       multi_stddev,
                'resid stddev': resid_and_stdev
                }.get(case)

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

    dataset = data_import.Dataset(plot_set)
    plotters = switch_plot(plot_fig)(dataset)

    plotters.update_layout(height=600,
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


    return [plotters]

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