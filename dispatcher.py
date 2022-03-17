
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask
from m200_dash import server as m200
from telonas2_dash import server as telo2

app = DispatcherMiddleware(m200, {
    '/prawler/m200': m200,
    '/prawler/telonas2': telo2
})