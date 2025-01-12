import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import pickle
import base64
import io
import calendar

# Inicializar la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Visualización de Modelos Portuarios"

# Cargar modelos y escaladores
modelo_regresor = joblib.load('modelo_ridge.pkl')
modelo_clasificador = joblib.load('modelo_rf_classifier.pkl')

scaler_reg = joblib.load('scaler_reg.pkl')
scaler_clf = joblib.load('scaler_clf.pkl')

with open('features_reg.pkl', 'rb') as f:
    features_reg = pickle.load(f)

with open('features_clf.pkl', 'rb') as f:
    features_clf = pickle.load(f)

with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)
    
nombres_meses_es = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
    7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}
# Layout principal
app.layout = html.Div(
    style={
        'backgroundImage': 'url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e")',
        'backgroundSize': 'cover',
        'backgroundRepeat': 'no-repeat',
        'backgroundPosition': 'center',
        'minHeight': '100vh',
        'padding': '20px'
    },
    children=[
        html.Div(
            style={
                'backgroundColor': 'rgba(255, 255, 255, 0.9)',
                'borderRadius': '10px',
                'padding': '20px',
                'maxWidth': '1200px',
                'margin': '0 auto'
            },
            children=[
                html.H1("Visualización de Predicciones Portuarias", style={'textAlign': 'center', 'color': '#003366'}),

                # Carga de datos
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Arrastra y suelta o ',
                            html.A('Selecciona un archivo CSV')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px 0'
                        },
                        multiple=False
                    ),
                ]),

                # Contenedor para filtros dinámicos
                html.Div(id='dynamic-filters'),

                # Resultados de predicción
                html.Div(id='output-data-upload', style={'marginTop': '20px'}),

                html.Hr(),

                # Pie de página
                html.Div([
                    html.P("Desarrollado para optimización portuaria - 2024 Grupo 2 IA", style={'textAlign': 'center', 'color': 'gray'})
                ])
            ]
        )
    ]
)

# Callback para procesar y predecir
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('dynamic-filters', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def process_and_predict(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Leer archivo CSV
            nuevos_datos = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            return html.Div([html.H5(f"Error al procesar el archivo: {e}")]), None

        # Preprocesar los datos
        nuevos_datos['Fecha_Llegada'] = pd.to_datetime(nuevos_datos['Fecha_Llegada'], format='%m/%d/%Y %H:%M', errors='coerce')
        nuevos_datos['Dia_Semana'] = nuevos_datos['Fecha_Llegada'].dt.dayofweek
        nuevos_datos['Mes'] = nuevos_datos['Fecha_Llegada'].dt.month
        nuevos_datos['Hora_Llegada'] = nuevos_datos['Fecha_Llegada'].dt.hour
        nuevos_datos['Minuto_Llegada'] = nuevos_datos['Fecha_Llegada'].dt.minute
        nuevos_datos['Eslora_Horas_Tardias'] = nuevos_datos['Eslora']

        # Codificar variables categóricas
        nuevos_datos_encoded = pd.get_dummies(nuevos_datos, columns=['Agencia', 'Procedencia'], drop_first=True)

        # Alinear las columnas de clasificación
        for col in features_clf:
            if col not in nuevos_datos_encoded.columns:
                nuevos_datos_encoded[col] = 0
        X_clf = nuevos_datos_encoded[features_clf]
        X_clf = scaler_clf.transform(X_clf)

        # Predicción del muelle
        nuevos_datos['Muelle Asignado'] = modelo_clasificador.predict(X_clf)
        nuevos_datos['Muelle Asignado'] = nuevos_datos['Muelle Asignado'].map(
            {i: class_names[i] for i in range(len(class_names))}
        )

        # Alinear las columnas de regresión
        for col in features_reg:
            if col not in nuevos_datos_encoded.columns:
                nuevos_datos_encoded[col] = 0
        X_reg = nuevos_datos_encoded[features_reg]
        X_reg = scaler_reg.transform(X_reg)

        # Predicción del retraso
        nuevos_datos['Retraso Estimado (Horas)'] = modelo_regresor.predict(X_reg)
        nuevos_datos['Retraso Estimado (Horas)'] = nuevos_datos['Retraso Estimado (Horas)'].round(2)

        # Crear columna de nombres de meses en español
        nuevos_datos['Mes_Llegada'] = nuevos_datos['Fecha_Llegada'].dt.month.map(nombres_meses_es)


        # Crear categorías de retrasos
        nuevos_datos['Categoría Retraso'] = pd.cut(
            nuevos_datos['Retraso Estimado (Horas)'],
            bins=[-np.inf, 0, 24, 72, np.inf],
            labels=['Sin retraso', 'Corto', 'Moderado', 'Largo']
        )

        # Generar opciones dinámicas para el filtro de muelle
        muelles_unicos = [{'label': muelle, 'value': muelle} for muelle in nuevos_datos['Muelle Asignado'].unique()]

        # Crear visualización
        fig_muelle = px.histogram(
            nuevos_datos,
            x='Muelle Asignado',
            title="Distribución de Predicciones por Muelle",
            labels={'Muelle Asignado': 'Tipo de Muelle'},
            color_discrete_sequence=['#003366']
        )

        fig_buques_mes = px.histogram(
            nuevos_datos,
            x='Mes_Llegada',
            color='Categoría Retraso',
            title="Número de Buques Esperados por Mes",
            labels={'Mes_Llegada': 'Mes', 'count': 'Número de Buques'},
            barmode='stack',
            category_orders={'Mes_Llegada': list(nombres_meses_es.values())}  # Ordenar los meses en español
        )

        # Crear filtros dinámicos y gráficos
        filtros = html.Div([
            dcc.Store(id='store-data', data=nuevos_datos.to_dict('records')),  # Guardar datos para filtros dinámicos
            dcc.Dropdown(
                id='filter-muelle',
                options=muelles_unicos,
                placeholder='Filtrar por Muelle',
                style={'marginBottom': '10px'}
            ),
            dcc.Input(
                id='filter-nombre-buque',
                type='text',
                placeholder='Buscar por Nombre del Buque',
                style={'marginBottom': '10px', 'width': '100%'}
            )
        ])

        tabla = dash_table.DataTable(
            id='table-results',
            columns=[
                {"name": "Registro", "id": "Registro"},
                {"name": "Nombre del Buque", "id": "Buque"},
                {"name": "Longitud del Buque (Eslora)", "id": "Eslora"},
                {"name": "Muelle Asignado", "id": "Muelle Asignado"},
                {"name": "Retraso Estimado (Horas)", "id": "Retraso Estimado (Horas)"},
                {"name": "Mes de Llegada", "id": "Mes_Llegada"},
                {"name": "Categoría Retraso", "id": "Categoría Retraso"}
            ],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': '#003366',
                'color': 'white',
                'fontWeight': 'bold'
            },
            data=[]  # Tabla inicial vacía
        )

        return html.Div([tabla, dcc.Graph(figure=fig_muelle), dcc.Graph(figure=fig_buques_mes)]), filtros

    return html.Div(["Por favor, subir un archivo CSV"]), None 


@app.callback(
    Output('table-results', 'data'),
    [
        Input('filter-muelle', 'value'),
        Input('filter-nombre-buque', 'value')
    ],
    [State('store-data', 'data')]
)
def filter_table(muelle_value, buque_value, data):
    # Verificar si los datos existen
    if data is None:
        return []

    # Convertir los datos almacenados en formato DataFrame
    df = pd.DataFrame(data)

    # Aplicar filtro de Muelle si está seleccionado y no es "Todos"
    if muelle_value and muelle_value != 'Todos':
        df = df[df['Muelle Asignado'] == muelle_value]

    # Aplicar filtro de Nombre del Buque si está proporcionado
    if buque_value:
        df = df[df['Buque'].str.contains(buque_value, case=False, na=False)]

    # Devolver los datos filtrados como una lista de diccionarios para la tabla
    return df.to_dict('records')


@app.callback(
    Output('filter-muelle', 'options'),
    Input('store-data', 'data')
)
def update_muelle_options(data):
    if data is None:
        return []
    df = pd.DataFrame(data)
    # Crear opciones dinámicas para el dropdown del filtro, incluyendo "Todos"
    opciones = [{'label': 'Todos', 'value': 'Todos'}]
    opciones += [{'label': muelle, 'value': muelle} for muelle in sorted(df['Muelle Asignado'].unique())]
    return opciones


if __name__ == '__main__':
    app.run_server(debug=True)
