from flask import(Flask, render_template, request, redirect, session)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

#Initialize
app = Flask(__name__)

@app.route('/', methods=('GET'))
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
