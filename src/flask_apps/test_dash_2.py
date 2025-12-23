from dash import Dash, dash_table
import mysql.connector

mydb = mysql.connector.connect(
host="localhost",
user="root",
password="Marathon#262",
database="OPTIONS"
)


mycursor = mydb.cursor() 
mycursor.execute("SELECT * FROM OPTIONS_DATA") 
db = mycursor.fetchall() 
                                 


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dash_table.DataTable(
    columns=[
        {"name": ["", "Year"], "id": "year" },
        {"name": ["City", "Montreal"], "id": "montreal", "deletable": [False, True]},
        {"name": ["City", "Toronto"], "id": "toronto", "renamable": True },
        {"name": ["City", "Ottawa"], "id": "ottawa", "hideable": "last"},
        {"name": ["City", "Vancouver"], "id": "vancouver"},
        {"name": ["Climate", "Temperature"], "id": "temp"},
        {"name": ["Climate", "Humidity"], "id": "humidity"},
    ],
    data=[
        {
            "year": row[0],
            "montreal": row[1],
            "toronto": row[2],
            "ottawa": row[3],
            "vancouver": row[4],
            "temp": row[5],
            "humidity": row[6],
        }
        for row in db
    ],
    export_format='csv',
    export_headers='display',
    merge_duplicate_headers=True
)


if __name__ == '__main__':
    app.run(debug=True)
