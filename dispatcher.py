
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask
from m200_dash import server as m200
from telonas2_dash import server as telo2
from co2_real_time import server as co2rt
from co2_validation import server as co2val
from landing_page import server as landing

app = DispatcherMiddleware(landing, {
    '/prawler/m200': m200,
    '/prawler/telonas2': telo2,
    '/co2/real_time': co2rt,
    '/co2/validation': co2val,
    '/main': landing#,
    #'/co2/validation': co2val
})