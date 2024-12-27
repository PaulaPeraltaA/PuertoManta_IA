import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import networkx as nx

# Inicializar la aplicación Dash
app = dash.Dash(__name__)
app.title = "Visualización de Modelos de Machine Learning"

# Estilos CSS para el fondo relacionado con el mar/portuario
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
                'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                'borderRadius': '10px',
                'padding': '20px',
                'maxWidth': '1200px',
                'margin': '0 auto'
            },
            children=[
                html.H1("Visualización de Modelos de Machine Learning", style={'textAlign': 'center', 'color': '#003366'}),
                
                # Sección de carga de datos
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
                
                # Mostrar un resumen de los datos
                html.Div(id='output-data-upload'),
                
                html.Hr(),
                
                # Sección de selección de variables y modelo
                html.Div([
                    html.Div([
                        html.Label("Selecciona la Variable Objetivo (Target):"),
                        dcc.Dropdown(id='target-variable', placeholder="Selecciona una variable")
                    ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20px'}),
                    
                    html.Div([
                        html.Label("Selecciona las Variables Predictoras (Features):"),
                        dcc.Dropdown(id='feature-variables', multi=True, placeholder="Selecciona una o más variables")
                    ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20px'}),
                ]),
                
                html.Div([
                    html.Label("Selecciona el Tipo de Modelo:"),
                    dcc.RadioItems(
                        id='model-type',
                        options=[
                            {'label': 'Regresión Lineal', 'value': 'linear'},
                            {'label': 'Árbol de Decisión (Regresión)', 'value': 'tree_reg'},
                            {'label': 'Árbol de Decisión (Clasificación)', 'value': 'tree_clf'}
                        ],
                        value='linear',
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    )
                ], style={'padding': '20px 40px'}),
                
                html.Button('Entrenar Modelo', id='train-button', n_clicks=0, style={'margin': '0 40px'}),
                
                html.Hr(),
                
                # Sección de resultados del modelo
                html.Div(id='model-output', style={'padding': '20px 40px'}),
                
                html.Hr(),
                
                # Pie de página
                html.Div([
                    html.P("Desarrollado por [Tu Nombre] - 2024", style={'textAlign': 'center', 'color': 'gray'})
                ])
            ]
        )
    ]
)

# Callback para procesar y mostrar los datos cargados
@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Leer el archivo CSV
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            return html.Div([
                html.H5(f"Error al procesar el archivo: {e}")
            ])
        
        # Mostrar las primeras filas
        return html.Div([
            html.H5(f"Archivo Cargado: {filename}", style={'color': '#003366'}),
            dash_table.DataTable(
                data=df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={
                    'height': 'auto',
                    'whiteSpace': 'normal',
                    'backgroundColor': 'rgba(255, 255, 255, 0.9)'
                },
                style_header={
                    'backgroundColor': '#003366',
                    'color': 'white',
                    'fontWeight': 'bold'
                }
            ),
            html.H6("Resumen Estadístico:", style={'color': '#003366'}),
            dash_table.DataTable(
                data=df.describe().reset_index().to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'height': 'auto',
                    'whiteSpace': 'normal',
                    'backgroundColor': 'rgba(255, 255, 255, 0.9)'
                },
                style_header={
                    'backgroundColor': '#003366',
                    'color': 'white',
                    'fontWeight': 'bold'
                }
            )
        ])

# Callback para actualizar las opciones de las variables objetivo y predictoras después de cargar el CSV
@app.callback(
    [
        Output('target-variable', 'options'),
        Output('feature-variables', 'options')
    ],
    [
        Input('upload-data', 'contents')
    ],
    [
        State('upload-data', 'filename')
    ]
)
def update_dropdowns(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Leer el archivo CSV
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            return [], []
        
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options
    else:
        return [], []

# Callback para entrenar el modelo y mostrar los resultados y predicciones
@app.callback(
    Output('model-output', 'children'),
    [
        Input('train-button', 'n_clicks')
    ],
    [
        State('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('target-variable', 'value'),
        State('feature-variables', 'value'),
        State('model-type', 'value')
    ]
)
def train_model(n_clicks, contents, filename, target, features, model_type):
    if n_clicks == 0:
        return ""
    
    if contents is None:
        return html.Div([
            html.H5("Por favor, sube un archivo CSV primero.", style={'color': '#003366'})
        ])
    
    if target is None or features is None or len(features) == 0:
        return html.Div([
            html.H5("Por favor, selecciona la variable objetivo y al menos una variable predictora.", style={'color': '#003366'})
        ])
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Leer el archivo CSV
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        return html.Div([
            html.H5(f"Error al procesar el archivo: {e}", style={'color': 'red'})
        ])
    
    # Separar características y etiquetas
    X = df[features]
    y = df[target]
    
    # Manejar variables categóricas si es clasificación
    if model_type == 'tree_clf':
        X = pd.get_dummies(X, drop_first=True)
        y = y.astype('category').cat.codes  # Convertir a códigos numéricos
    
    # Crear columnas ETD_Horas y Fecha_Zarpe_Horas si no existen
    if 'ETD_Horas' not in df.columns or 'Fecha_Zarpe_Horas' not in df.columns:
        if 'Fecha_Zarpe_Estimada' in df.columns and 'Fecha_Zarpe' in df.columns:
            # Asegurarse de que las columnas están en formato datetime
            df['Fecha_Zarpe_Estimada'] = pd.to_datetime(df['Fecha_Zarpe_Estimada'], errors='coerce')
            df['Fecha_Zarpe'] = pd.to_datetime(df['Fecha_Zarpe'], errors='coerce')
            
            df['ETD_Horas'] = df['Fecha_Zarpe_Estimada'].dt.hour + \
                              df['Fecha_Zarpe_Estimada'].dt.minute / 60 + \
                              df['Fecha_Zarpe_Estimada'].dt.second / 3600
            df['Fecha_Zarpe_Horas'] = df['Fecha_Zarpe'].dt.hour + \
                                        df['Fecha_Zarpe'].dt.minute / 60 + \
                                        df['Fecha_Zarpe'].dt.second / 3600
        else:
            return html.Div([
                html.H5("Las columnas 'Fecha_Zarpe_Estimada' y/o 'Fecha_Zarpe' no existen en el DataFrame.", style={'color': 'red'})
            ])
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar el modelo según el tipo seleccionado
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Crear gráfica de valores reales vs predichos con Plotly
        fig = px.scatter(
            x=y_test, y=y_pred,
            labels={'x': 'Valores Reales', 'y': 'Valores Predichos'},
            title='Regresión Lineal: Valores Reales vs Predichos',
            trendline='ols',
            opacity=0.7
        )
        fig.update_layout(template='plotly_dark')
        
        # Crear tabla de predicciones
        predictions = pd.DataFrame({
            'Valores Reales': y_test,
            'Predicciones': y_pred
        }).reset_index(drop=True)
        
        predictions_table = dash_table.DataTable(
            data=predictions.head(20).to_dict('records'),  # Mostrar las primeras 20 predicciones
            columns=[{"name": i, "id": i} for i in predictions.columns],
            page_size=20,
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'whiteSpace': 'normal',
                'backgroundColor': 'rgba(255, 255, 255, 0.9)'
            },
            style_header={
                'backgroundColor': '#003366',
                'color': 'white',
                'fontWeight': 'bold'
            }
        )
        
        return html.Div([
            html.H3("Resultados de Regresión Lineal", style={'color': '#003366'}),
            html.P(f"**RMSE:** {rmse:.2f} horas", style={'color': '#003366'}),
            html.P(f"**R²:** {r2:.2f}", style={'color': '#003366'}),
            dcc.Graph(figure=fig),
            html.H4("Predicciones (Primeras 20 Instancias):", style={'color': '#003366'}),
            predictions_table
        ])
    
    elif model_type == 'tree_reg':
        model = DecisionTreeRegressor(max_depth=7, min_samples_split=30, min_samples_leaf=15, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Visualizar el árbol de decisión usando Plotly y NetworkX
        tree_text = export_text(model, feature_names=features)
        tree_graph = nx.DiGraph()
        
        # Parsear el texto del árbol para crear un grafo
        lines = tree_text.split('\n')
        parent = None
        stack = []
        for line in lines:
            if not line.strip():
                continue
            depth = line.count('|   ')
            content = line.strip().replace('|--- ', '')
            node_id = len(tree_graph.nodes)
            label = content
            tree_graph.add_node(node_id, label=label)
            if depth > 0:
                # Asignar el padre basándose en la profundidad
                parent = stack[depth - 1]
                tree_graph.add_edge(parent, node_id)
            if len(stack) > depth:
                stack[depth] = node_id
            else:
                stack.append(node_id)
        
        # Posicionar los nodos usando el layout de Tree
        pos = nx.nx_pydot.graphviz_layout(tree_graph, prog='dot')
        edge_x = []
        edge_y = []
        for edge in tree_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        for node in tree_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(tree_graph.nodes[node]['label'])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="bottom center",
            marker=dict(
                size=20,
                color='#003366',
                line_width=2
            ),
            hoverinfo='text'
        )
        
        fig_tree = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                 showlegend=False,
                                 hovermode='closest',
                                 margin=dict(b=20,l=5,r=5,t=40),
                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 title='Árbol de Decisión para Regresión'
                             ))
        
        fig_tree.update_layout(template='plotly_dark')
        
        # Crear gráfica de valores reales vs predichos con Plotly
        fig_scatter = px.scatter(
            x=y_test, y=y_pred,
            labels={'x': 'Valores Reales', 'y': 'Valores Predichos'},
            title='Árbol de Decisión (Regresión): Valores Reales vs Predichos',
            trendline='ols',
            opacity=0.7
        )
        fig_scatter.update_layout(template='plotly_dark')
        
        # Crear tabla de predicciones
        predictions = pd.DataFrame({
            'Valores Reales': y_test,
            'Predicciones': y_pred
        }).reset_index(drop=True)
        
        predictions_table = dash_table.DataTable(
            data=predictions.head(20).to_dict('records'),  # Mostrar las primeras 20 predicciones
            columns=[{"name": i, "id": i} for i in predictions.columns],
            page_size=20,
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'whiteSpace': 'normal',
                'backgroundColor': 'rgba(255, 255, 255, 0.9)'
            },
            style_header={
                'backgroundColor': '#003366',
                'color': 'white',
                'fontWeight': 'bold'
            }
        )
        
        return html.Div([
            html.H3("Resultados de Árbol de Decisión (Regresión)", style={'color': '#003366'}),
            html.P(f"**RMSE:** {rmse:.2f} horas", style={'color': '#003366'}),
            html.P(f"**R²:** {r2:.2f}", style={'color': '#003366'}),
            dcc.Graph(figure=fig_tree),
            dcc.Graph(figure=fig_scatter),
            html.H4("Predicciones (Primeras 20 Instancias):", style={'color': '#003366'}),
            predictions_table
        ])
    
    elif model_type == 'tree_clf':
        model = DecisionTreeClassifier(max_depth=7, min_samples_split=30, min_samples_leaf=15, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=df[target].astype(str).unique(), zero_division=0, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        
        # Crear gráfica de la matriz de confusión con Plotly
        fig_conf = px.imshow(
            conf_matrix,
            labels=dict(x="Predicción", y="Real", color="Cantidad"),
            x=df[target].astype(str).unique(),
            y=df[target].astype(str).unique(),
            text_auto=True,
            color_continuous_scale='Blues',
            title="Matriz de Confusión"
        )
        fig_conf.update_layout(template='plotly_dark')
        
        # Crear tabla de reporte de clasificación
        report_table = dash_table.DataTable(
            data=class_report_df.reset_index().to_dict('records'),
            columns=[{"name": i, "id": i} for i in class_report_df.reset_index().columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'whiteSpace': 'normal',
                'backgroundColor': 'rgba(255, 255, 255, 0.9)'
            },
            style_header={
                'backgroundColor': '#003366',
                'color': 'white',
                'fontWeight': 'bold'
            }
        )
        
        # Crear tabla de predicciones
        predictions = pd.DataFrame({
            'Clase Real': y_test,
            'Clase Predicha': y_pred
        }).reset_index(drop=True)
        
        predictions_table = dash_table.DataTable(
            data=predictions.head(20).to_dict('records'),  # Mostrar las primeras 20 predicciones
            columns=[{"name": i, "id": i} for i in predictions.columns],
            page_size=20,
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'whiteSpace': 'normal',
                'backgroundColor': 'rgba(255, 255, 255, 0.9)'
            },
            style_header={
                'backgroundColor': '#003366',
                'color': 'white',
                'fontWeight': 'bold'
            }
        )
        
        return html.Div([
            html.H3("Resultados de Árbol de Decisión (Clasificación)", style={'color': '#003366'}),
            html.P(f"**Precisión (Accuracy):** {accuracy:.2f}", style={'color': '#003366'}),
            html.H4("Matriz de Confusión:", style={'color': '#003366'}),
            dcc.Graph(figure=fig_conf),
            html.H4("Reporte de Clasificación:", style={'color': '#003366'}),
            report_table,
            html.H4("Predicciones (Primeras 20 Instancias):", style={'color': '#003366'}),
            predictions_table
        ])
    
    else:
        return html.Div([
            html.H5("Tipo de modelo no soportado.", style={'color': 'red'})
        ])

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
 