import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ------------------------------
# 1. Carga y Exploración Inicial de Datos
# ------------------------------
# Cargar la base de datos desde el CSV
ruta_csv = "C:/Users/Stephany/Desktop/Nueva carpeta (3)/PuertoManta_IA/resultado_con_fecha_zarpe.csv"  # Reemplaza esto con tu ruta
df = pd.read_csv(ruta_csv, thousands=',')

# Verificar el DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Revisar información general
print("\nInformación general del DataFrame:")
print(df.info())

# Revisar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# ------------------------------
# 2. Corrección del Cálculo de 'Error_Horas'
# ------------------------------
# Asegurar que 'Fecha_Llegada' esté en formato datetime
df['Fecha_Llegada'] = pd.to_datetime(df['Fecha_Llegada'], format='%m/%d/%Y %H:%M', errors='coerce')

# Extraer fecha de 'Fecha_Llegada' para combinar con 'ETD'
df['Fecha'] = df['Fecha_Llegada'].dt.date

# Función para parsear 'ETD' correctamente
def parse_etd(etd_str):
    try:
        # Intentar parsear con segundos
        return datetime.strptime(etd_str, '%H:%M:%S').time()
    except ValueError:
        try:
            # Intentar sin segundos
            return datetime.strptime(etd_str, '%H:%M').time()
        except ValueError:
            # Asignar hora media si falla
            return datetime.strptime('12:00', '%H:%M').time()

df['ETD_Time'] = df['ETD'].apply(parse_etd)

# Crear 'Fecha_Zarpe_Estimada' combinando 'Fecha' y 'ETD_Time'
df['Fecha_Zarpe_Estimada'] = df.apply(lambda row: datetime.combine(row['Fecha'], row['ETD_Time']), axis=1)

# Convertir 'Fecha_Zarpe' a datetime
df['Fecha_Zarpe'] = pd.to_datetime(df['Fecha_Zarpe'], format='%m/%d/%Y %H:%M', errors='coerce')

# Calcular 'Error_Horas' como diferencia entre 'Fecha_Zarpe' y 'Fecha_Zarpe_Estimada'
df['Error_Horas'] = (df['Fecha_Zarpe'] - df['Fecha_Zarpe_Estimada']).dt.total_seconds() / 3600

# Mostrar distribución de 'Error_Horas'
print("\nDistribución de 'Error_Horas':")
print(df['Error_Horas'].describe())

# ------------------------------
# 3. Eliminación de Filas con Valores Nulos y Outliers
# ------------------------------
# Eliminar filas con valores nulos en las columnas relevantes
df = df.dropna(subset=['Eslora', 'Horas_Tardias', 'Error_Horas', 'Fecha_Zarpe'])

# Opcional: eliminar filas donde 'Error_Horas' es extremadamente alto o bajo
# Esto puede ayudar a mejorar la calidad del modelo
df = df[(df['Error_Horas'] > -100) & (df['Error_Horas'] < 1000)]  # Ajusta según tu conocimiento del dominio

# ------------------------------
# 4. Limpieza Adicional y Feature Engineering
# ------------------------------
# Identificar y eliminar columnas con poca o ninguna variabilidad
zero_variance_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"\nColumnas con única categoría o valor (sin variabilidad): {zero_variance_cols}")
df = df.drop(columns=zero_variance_cols)
print(f"Tamaño del conjunto de datos después de eliminar columnas sin variabilidad: {df.shape}")

# Revisar nuevamente valores nulos después de eliminación
print("\nValores nulos por columna después de limpieza:")
print(df.isnull().sum())

# Reemplazar 'IMO' = 0 con NaN y eliminar esas filas
df['IMO'] = df['IMO'].replace(0, np.nan)
df = df.dropna(subset=['IMO'])
df['IMO'] = df['IMO'].astype(int)

# Convertir variables categóricas a tipo 'category'
categorical_cols = ['Agencia', 'Muelle', 'Procedencia']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Codificación de variables categóricas utilizando One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"\nConjunto de datos después de codificar variables categóricas: {df_encoded.shape}")

# ------------------------------
# 5. Definición de Variables para Modelos
# ------------------------------
# 5.1. Regresión Lineal para predecir 'Error_Horas'
# ------------------------------
# Seleccionar características para regresión
features_reg = ['Eslora', 'Horas_Tardias', 'IMO'] + \
              [col for col in df_encoded.columns if col.startswith('Agencia_') or col.startswith('Procedencia_')]

X_reg = df_encoded[features_reg]
y_reg = df_encoded['Error_Horas']

# 5.2. Clasificación para predecir 'Muelle'
# ------------------------------
# Definir variables predictoras y objetivo para clasificación
features_clf = ['Eslora', 'Horas_Tardias', 'IMO'] + \
              [col for col in df_encoded.columns if col.startswith('Agencia_') or col.startswith('Procedencia_')]

X_clf = df_encoded[features_clf]
y_clf = df['Muelle'].cat.codes  # Usar codificación categórica para 'Muelle'

# ------------------------------
# 6. División de Datos en Entrenamiento y Prueba
# ------------------------------
# División para Regresión
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# División para Clasificación
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# ------------------------------
# 7. Balanceo de Clases para Clasificación
# ------------------------------
# Verificar la distribución de clases antes del balanceo
print("\nDistribución de clases antes del balanceo:")
print(y_train_c.value_counts())

# Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_c_balanced, y_train_c_balanced = smote.fit_resample(X_train_c, y_train_c)

# Verificar la distribución de clases después del balanceo
print("\nDistribución de clases después del balanceo:")
print(pd.Series(y_train_c_balanced).value_counts())

# ------------------------------
# 8. Escalado de Características
# ------------------------------
# Escalar características para regresión
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Escalar características para clasificación
scaler_clf = StandardScaler()
X_train_c_balanced_scaled = scaler_clf.fit_transform(X_train_c_balanced)
X_test_c_scaled = scaler_clf.transform(X_test_c)

# ------------------------------
# 9. Implementación de Modelos
# ------------------------------
# 9.1. Regresión Lineal Múltiple
# ------------------------------
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg = modelo_regresion.predict(X_test_reg_scaled)

# Evaluación del modelo de regresión
rmse_reg = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2_reg = r2_score(y_test_reg, y_pred_reg)

print(f"\nRegresión Lineal Múltiple - RMSE: {rmse_reg:.2f} horas")
print(f"Regresión Lineal Múltiple - R²: {r2_reg:.2f}")

# Visualización: Valores reales vs predichos para regresión
plt.figure(figsize=(8,6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel("Valores Reales (Error_Horas)")
plt.ylabel("Valores Predichos (Error_Horas)")
plt.title("Regresión Lineal Múltiple: Valores reales vs predichos")
plt.grid(True)
plt.show()

# 9.2. Árbol de Decisión para Clasificación
# ------------------------------
modelo_arbol = DecisionTreeClassifier(
    max_depth=10, 
    min_samples_split=20, 
    min_samples_leaf=10, 
    random_state=42
)
modelo_arbol.fit(X_train_c_balanced_scaled, y_train_c_balanced)
y_pred_c = modelo_arbol.predict(X_test_c_scaled)

# Evaluación del modelo de clasificación
accuracy = accuracy_score(y_test_c, y_pred_c)
print(f"\nPrecisión del Árbol de Decisión: {accuracy:.2f}")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_c, y_pred_c))

print("\nReporte de Clasificación:")
print(classification_report(y_test_c, y_pred_c, target_names=df['Muelle'].cat.categories.astype(str)))

# Visualización del árbol de decisión
plt.figure(figsize=(15, 10))
plot_tree(
    modelo_arbol, 
    feature_names=features_clf, 
    class_names=df['Muelle'].cat.categories.astype(str), 
    filled=True, 
    fontsize=12
)
plt.title("Árbol de Decisión para Clasificación de Muelle")
plt.show()

# ------------------------------
# 10. Comparativa Visual entre ETD y Fecha de Zarpe
# ------------------------------
plt.figure(figsize=(14,7))
# Convertir 'Fecha_Zarpe_Estimada' y 'Fecha_Zarpe' a horas del día
df['ETD_Horas'] = df['Fecha_Zarpe_Estimada'].dt.hour + \
                  df['Fecha_Zarpe_Estimada'].dt.minute / 60 + \
                  df['Fecha_Zarpe_Estimada'].dt.second / 3600
df['Fecha_Zarpe_Horas'] = df['Fecha_Zarpe'].dt.hour + \
                            df['Fecha_Zarpe'].dt.minute / 60 + \
                            df['Fecha_Zarpe'].dt.second / 3600

plt.scatter(df.index, df['ETD_Horas'], label='ETD', alpha=0.5, marker='o')
plt.scatter(df.index, df['Fecha_Zarpe_Horas'], label='Fecha Zarpe', alpha=0.5, marker='x')
plt.xlabel("Índice de Buques")
plt.ylabel("Hora del Día (Horas)")
plt.title("Comparación entre ETD y Hora Real de Zarpe")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 11. Identificación de Retrasos Significativos
# ------------------------------
umbral_retraso = 30
retrasos_significativos = df[df['Error_Horas'] > umbral_retraso]

print(f"\nNúmero de embarcaciones con más de {umbral_retraso} horas de retraso: {retrasos_significativos.shape[0]}")

# Visualización de Retrasos Significativos
plt.figure(figsize=(10,6))
plt.hist(retrasos_significativos['Error_Horas'], bins=50, color='red', alpha=0.7)
plt.xlabel('Retraso (Horas)')
plt.ylabel('Cantidad de Embarcaciones')
plt.title(f'Embarcaciones con más de {umbral_retraso} Horas de Retraso')
plt.show()

