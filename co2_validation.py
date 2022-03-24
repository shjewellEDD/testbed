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

set_url = 'https://dunkel.pmel.noaa.gov:9290/erddap/tabledap/asvco2_gas_validation_summary_mirror.csv'
#set_url =

custom_sets = [{'label': 'XCO2 Mean',       'value': 'resids'},
               {'label': 'XCO2 Residuals',  'value': 'cals'},
               {'label': 'XCO2 Delta',      'value': 'temp resids'},
               {'label': 'CO2 Pres. Mean',  'value': 'stddev'},
               {'label': 'CO2 Mean',        'value': 'resid stddev'},
        ]

dataset = data_import.Dataset(set_url)

colors = {'background': '#111111', 'text': '#7FDBFF'}

app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                external_stylesheets=[dbc.themes.SLATE])
server = app.server

tools_card = dbc.Card([
    dbc.CardBody(
           style={'backgroundColor': colors['background']},
           children=[
                dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=dataset.t_start,
                max_date_allowed=dataset.t_end,
                start_date=dataset.t_end - datetime.timedelta(days=14),
                end_date=dataset.t_end
                ),
            dhtml.Label(['Select Set']),
            dcc.Dropdown(
                id="select_x",
                options=custom_sets,
                value='resids',
                clearable=False
                ),
           dhtml.Label(['']),
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
                dbc.Col(tools_card, width=3),
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
    [Output('graphs', 'figure'),
     Output('datatable', 'data'),
     Output('datatable', 'columns')],
    [Input('select_x', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')
     ])

def load_plot(plot_fig, start_date, end_date):

    def off_ref(df):
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

        epoff = df[df['INSTRUMENT_STATE'] == 'EPOFF']
        apoff = df[df['INSTRUMENT_STATE'] == 'APOFF']

        load_plots = make_subplots(rows=1, cols=1,
                                   subplot_titles=['Pressure'],
                                   shared_yaxes=False)

        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_RESIDUAL_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=epoff['CO2_REF_LAB'], y=epoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='EPOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_RESIDUAL_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)
        load_plots.add_scatter(x=apoff['CO2_REF_LAB'], y=apoff['CO2_DRY_TCORR_RESIDUAL_MEAN_ASVCO2'], name='APOFF', hoverinfo='x+y+name',
                               mode='markers', marker={'size': 5}, row=1, col=1)

        load_plots['layout'].update(height=600,
                                    title=' ',
                                    hovermode='x unified',
                                    xaxis_showticklabels=True,
                                    yaxis_fixedrange=True,
                                    yaxis_title='Residual',
                                    xaxis_title='CO2 Gas Concentration',
                                    xaxis=dict(showgrid=False),
                                    showlegend=False, modebar={'orientation': 'h'}, autosize=True,
                                    margin=dict(l=25, r=25, b=25, t=25, pad=4)
                                    )

        #dtable = dash_table.DataTable()

        columns = [{'id': 'state', 'name': 'State'},
                 {'id': 'size', 'name': 'Size'}]

        table_df = pd.concat([epoff, apoff])
        drivers = [list(table_df['INSTRUMENT_STATE'].unique()), list(table_df.groupby('INSTRUMENT_STATE').size())]
        sizes = [str(x[0]) + ", " + str(x[1]) for x in drivers]

        table_data = [{'state': list(table_df['INSTRUMENT_STATE'].unique())},
                             {'size': sizes}]

        return load_plots, table_data, columns


    def cal_ref(df):
        '''
        Select serial and date
            serial: SN_ASVCO2
            date: time
        Residual for ZPPCAL and SPPCAL vs gas concentration
            state: INSTRUMENT_STATE
            gas:
        :return:
        '''


        return


    def multi_ref(df):
        '''
        Select serial, LICOR firmware, ASVCO2 firmware, date range
        Boolean temperature correct residual
        Residual
        :return:
        '''

        return

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

    def switch_plot(case, data):
        return {'resids':       off_ref(data),
                'cals':         cal_ref(data),
                'temp resids':  multi_ref(data),
                'stddev':       multi_stddev(data),
                'resid stddev': resid_and_stdev(data)
                }.get(case)

    states = ['ZPON', 'ZPOFF', 'ZPPCAL', 'SPON', 'SPOFF', 'SPPCAL', 'EPON', 'EPOFF', 'APON', 'APOFF']

    data = dataset.ret_data(t_start=start_date, t_end=end_date)
    print(plot_fig)
    plotters = switch_plot(plot_fig, data)

    plotters[0].update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        autosize=True
    )

    return plotters

if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', port=8050, debug=True)

    app.run_server(debug=True)