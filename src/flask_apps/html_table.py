import dash
from dash import dash_table
import pandas as pd

app = dash.Dash(__name__)

df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})

app.layout = dash_table.DataTable(
    id='table',
    data=df.to_dict('records'),
    columns=[{'name': i, 'id': i} for i in df.columns],
    editable=True
)

if __name__ == '__main__':
    app.run_server(debug=True)