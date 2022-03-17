import m200_dash as m200
import telonas2_dash as telo2
from flask import Flask

app = Flask(__name__)

@app.route('/prawler/m200')
def m200_dash():
    return m200


@app.route('/prawler/telo2')
def telonas2_dash():
    return telo2

if __name__ == '__main__':
    app('localhost', 8080, app, use_reloader=True)