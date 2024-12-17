import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

# Crear app Dash
app = dash.Dash(__name__)

# Layout de la app
app.layout = html.Div([
    html.H1("Visualización de Modelos"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Subir archivo CSV'),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.H2("Modelo"),
    dcc.Dropdown(
        id='model-selector',
        options=[
            {'label': 'Árbol de Decisión', 'value': 'tree'},
            {'label': 'Regresión Lineal', 'value': 'linear'}
        ],
        value='tree'
    ),
    html.Div(id='model-output'),
])

# Callback para cargar datos y entrenar modelos
@app.callback(
    Output('model-output', 'children'),
    Input('upload-data', 'contents'),
    Input('model-selector', 'value')
)
def train_model(contents, model_type):
    if contents is None:
        return "Sube un archivo para entrenar el modelo."

    # Leer CSV
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Separar características y etiquetas (supongamos que son las columnas X, Y)
    X = df[['X']]  # Cambiar por nombres reales
    y = df['Y']    # Cambiar por nombres reales

    # Entrenar modelos
    if model_type == 'tree':
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X, y)

        # Visualizar árbol
        plt.figure(figsize=(10, 8))
        plot_tree(model, feature_names=['X'], filled=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return html.Img(src='data:image/png;base64,{}'.format(encoded_image))

    elif model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)

        # Graficar línea de regresión
        plt.scatter(X, y, color='blue')
        plt.plot(X, model.predict(X), color='red')
        plt.title('Regresión Lineal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return html.Img(src='data:image/png;base64,{}'.format(encoded_image))

# Ejecutar la app
if __name__ == '__main__':
    app.run_server(debug=True)
