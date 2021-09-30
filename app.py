from flask import(Flask, render_template, request, redirect, session)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

#Initialize
app = Flask(__name__)

#Query code
username = 'kzblgxklbcxgjk'
password = '42cbe5c9d571af401f937b7d8b23ff8b49cca031c697b2ad2b222f36fa48dfca'
host = 'ec2-34-199-209-37.compute-1.amazonaws.com'
db = 'dd351di0els2o7'

engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}/{db}')

# @app.route('/', methods=['POST','GET'])
# def home():
#     df = pd.read_sql_query(f"SELECT * FROM rankings ORDER_BY(week, rank)")
#     return render_template('index2.html', tables = [df.to_html(classes='data')], titles = df.columns.values)

if __name__ == '__main__':
    app.run()
