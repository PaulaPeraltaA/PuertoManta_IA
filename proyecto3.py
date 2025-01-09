import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar la base de datos desde el CSV
###ruta_csv = 'resultado_con_fecha_zarpe.csv'  # Reemplaza esto con tu ruta
ruta_csv = "C:/Users/Stephany/Desktop/Nueva carpeta (3)/PuertoManta_IA/resultado_con_fecha_zarpe.csv"

df = pd.read_csv(ruta_csv)

# Verificar el DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Revisar información general
print("\nInformación general del DataFrame:")
print(df.info())

# Revisar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Asegurar que las fechas estén en formato datetime
df['ETD'] = pd.to_datetime(df['ETD'], errors='coerce')
df['Fecha_Zarpe'] = pd.to_datetime(df['Fecha_Zarpe'], errors='coerce')

# Calcular la columna 'Error_Horas'
df['Error_Horas'] = (df['Fecha_Zarpe'] - df['ETD']).dt.total_seconds() / 3600
print("\nColumnas después de calcular Error_Horas:")
print(df.head())

# Eliminar filas con valores nulos en las columnas relevantes
df = df.dropna(subset=['Eslora', 'Horas_Tardias', 'Error_Horas'])

# Variables predictoras (X) y objetivo (y)
X = df[['Eslora', 'Horas_Tardias']]
y = df['Error_Horas']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

# Predicciones
y_pred = modelo_regresion.predict(X_test)

# Evaluar el modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualización: Valores reales vs predichos
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Regresión Lineal: Valores reales vs predichos")
plt.grid()
plt.show()

# --- Modelo de Clasificación (Árbol de Decisión) ---

# Convertir 'Muelle' a valores numéricos
df['Muelle'] = df['Muelle'].astype('category').cat.codes

# Variables predictoras y objetivo para clasificación
X_clf = df[['Horas_Tardias', 'Eslora']]
y_clf = df['Muelle']

# Dividir los datos
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Crear el modelo de árbol de decisión
modelo_arbol = DecisionTreeClassifier(max_depth=4)
modelo_arbol.fit(X_train_c, y_train_c)

# Predicciones
y_pred_c = modelo_arbol.predict(X_test_c)

# Evaluación
accuracy = accuracy_score(y_test_c, y_pred_c)
print(f"Precisión del Árbol de Decisión: {accuracy:.2f}")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_c, y_pred_c))

# Visualización del árbol de decisión
plt.figure(figsize=(15, 10))
plot_tree(modelo_arbol, feature_names=X_clf.columns, class_names=True, filled=True)
plt.title("Árbol de Decisión")
plt.show()

# --- Comparación visual entre ETD y Fecha de Zarpe ---

# Eliminar valores nulos antes de graficar
df_fechas = df.dropna(subset=['ETD', 'Fecha_Zarpe'])

# Crear el gráfico
plt.plot(df_fechas['ETD'], label='ETD', linestyle='--', marker='o')
plt.plot(df_fechas['Fecha_Zarpe'], label='Fecha Zarpe', linestyle='-', marker='x')
plt.xlabel("Índice de Buques")
plt.ylabel("Fecha y Hora")
plt.title("Comparación entre ETD y Hora Real de Zarpe")
plt.legend()
plt.grid()
plt.show()
