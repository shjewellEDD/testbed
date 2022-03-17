import m200_dash as m200
import telonas2_dash as telo2
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask
#import flask_app
import master_app

app = DispatcherMiddleware(master_app, {
    '/prawler/m200': m200.server,
    '/prawler/telonas2': telo2.server
})