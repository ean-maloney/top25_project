from flask import(Flask, render_template, request, redirect, session)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

#Initialize
app = Flask(__name__)

#Query code
username = 'godenmyqjrmzoe'
password = '673e2f643ed4ddbe58c111219261e6872457e45e34e34c617efa029f10429c0f'
host = 'ec2-54-83-137-206.compute-1.amazonaws.com'
db = 'dc995n2umb789o'

@app.route('/', methods=['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/index2.html', methods=['POST','GET'])
def models():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run()
