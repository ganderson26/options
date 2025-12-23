from flask import Flask, render_template, redirect, url_for, request

import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import *
import mysql.connector

mydb = mysql.connector.connect(
host="localhost",
user="root",
password="Marathon#262",
database="OPTIONS"
)

app = Flask(__name__)

@app.route("/") 
def productdetails(): 
    try: 
        mycursor = mydb.cursor() 
        mycursor.execute("SELECT * FROM OPTIONS_DATA") 
        db = mycursor.fetchall() 
        return render_template("view_transactions.html", dbhtml = db)                                   
    except Exception as e: 
        return(str(e))

if __name__=="main": 
    app.run(debug=True)
