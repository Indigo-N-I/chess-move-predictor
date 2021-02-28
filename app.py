import os

from flask import FLask, jsonify
from flask import url_for

from dtenv import load_dotenv

from lichess_id import ID, SECRET

from authlib.integrations.flask_client import OAuth

app = Flask(__name__)
app.secret_key = SECRET
app.config['LICHESS_CLIENT_ID'] = ID
app.config['LICHESS_CLIENT_SECRET'] = SECRET

app.config['LICHESS_ACCESS_TOKEN_URL'] = 'http://127.0.0.1:5000/'
app.config['LICHESS_AUTHORIZE_URL'] = 'http://127.0.0.1:5000/authorize'

@app.route('/')
def login
