from werkzeug.serving import run_simple
from run import app

if __name__ == '__main__':
    run_simple('localhost', 8080, app,
               use_reloader=True,
               use_debugger=True,
               use_evalex=True)