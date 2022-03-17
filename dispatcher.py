
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask

from m200_dash import app as m200
from telonas2_dash import app as telo2

app = DispatcherMiddleware(m200, {
    '/prawler/m200': m200,
    '/prawler/telonas2': telo2
})