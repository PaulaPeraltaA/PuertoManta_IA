import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer 
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
# ------------------------------
# 1. Carga y Exploración Inicial de Datos
# ------------------------------
# Cargar la base de datos desde el CSV
ruta_csv = 'resultado_con_fecha_zarpe.csv'  # Reemplaza esto con tu ruta
df = pd.read_csv(ruta_csv, thousands=',')

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
        return pd.to_datetime(etd_str, format='%H:%M:%S').time()
    except ValueError:
        try:
            return pd.to_datetime(etd_str, format='%H:%M').time()
        except ValueError:
            return pd.to_datetime('12:00', format='%H:%M').time()

df['ETD_Time'] = df['ETD'].apply(parse_etd)

# Crear 'Fecha_Zarpe_Estimada' combinando 'Fecha' y 'ETD_Time'
df['Fecha_Zarpe_Estimada'] = df.apply(lambda row: pd.to_datetime(row['Fecha']).replace(
    hour=row['ETD_Time'].hour,
    minute=row['ETD_Time'].minute,
    second=row['ETD_Time'].second
), axis=1)

# Convertir 'Fecha_Zarpe' a datetime
df['Fecha_Zarpe'] = pd.to_datetime(df['Fecha_Zarpe'], format='%m/%d/%Y %H:%M', errors='coerce')

# Calcular 'Error_Horas' como diferencia entre 'Fecha_Zarpe' y 'Fecha_Zarpe_Estimada'
df['Error_Horas'] = (df['Fecha_Zarpe'] - df['Fecha_Zarpe_Estimada']).dt.total_seconds() / 3600

# ------------------------------
# 3. Eliminación de Filas con Valores Nulos y Outliers
# ------------------------------
df = df.dropna(subset=['Eslora', 'Error_Horas', 'Fecha_Zarpe'])

# Eliminación basada en percentiles para manejar outliers extremos
lower_percentile = df['Error_Horas'].quantile(0.01)
upper_percentile = df['Error_Horas'].quantile(0.99)
df = df[(df['Error_Horas'] >= lower_percentile) & (df['Error_Horas'] <= upper_percentile)]

# ------------------------------
# 4. Limpieza Adicional y Feature Engineering
# ------------------------------
# Eliminar columnas con poca o ninguna variabilidad
zero_variance_cols = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=zero_variance_cols)

# Reemplazar 'IMO' = 0 con NaN y eliminar esas filas
df['IMO'] = df['IMO'].replace(0, np.nan)
df = df.dropna(subset=['IMO'])
df['IMO'] = df['IMO'].astype(int)

# Convertir variables categóricas a tipo 'category'
categorical_cols = ['Agencia', 'Muelle', 'Procedencia']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Agregar variables temporales
df['Dia_Semana'] = df['Fecha_Llegada'].dt.dayofweek
df['Mes'] = df['Fecha_Llegada'].dt.month
df['Hora_Llegada'] = df['Fecha_Llegada'].dt.hour
df['Minuto_Llegada'] = df['Fecha_Llegada'].dt.minute

# Crear interacciones
# (Asegúrate de que 'Horas_Tardias' exista en el DataFrame)
if 'Horas_Tardias' in df.columns:
    df['Eslora_Horas_Tardias'] = df['Eslora'] * df['Horas_Tardias']
    df = df.drop(columns=['Horas_Tardias'])
else:
    df['Eslora_Horas_Tardias'] = df['Eslora']  # O ajusta según corresponda

# Codificación de variables categóricas utilizando One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Agencia', 'Procedencia'], drop_first=True)

# ------------------------------
# 5. Definición de Variables para Modelos
# ------------------------------
# Regresión Lineal para predecir 'Error_Horas'
features_reg = ['Eslora', 'IMO', 'Dia_Semana', 'Mes', 'Hora_Llegada', 'Minuto_Llegada', 'Eslora_Horas_Tardias'] + \
              [col for col in df_encoded.columns if col.startswith('Agencia_') or col.startswith('Procedencia_')]

X_reg = df_encoded[features_reg]
y_reg = df['Error_Horas']

# Clasificación para predecir 'Muelle'
features_clf = ['Eslora', 'IMO', 'Dia_Semana', 'Mes', 'Hora_Llegada', 'Minuto_Llegada', 'Eslora_Horas_Tardias'] + \
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
# 7. Escalado de Características
# ------------------------------
# Escalar características para regresión
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Escalar características para clasificación
scaler_clf = StandardScaler()
X_train_c_scaled = scaler_clf.fit_transform(X_train_c)
X_test_c_scaled = scaler_clf.transform(X_test_c)


##random forest con parametros mejores:
param_grid_rf = {
    'n_estimators': [50, 100, 200],      # Número de árboles
    'max_depth': [5, 10, 15],           # Profundidad máxima
    'min_samples_split': [10, 20, 30],  # Tamaño mínimo para dividir
    'min_samples_leaf': [5, 10, 15]     # Tamaño mínimo de hojas
}

grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1
)

grid_search_rf.fit(X_train_reg_scaled, y_train_reg)

# Obtener los mejores parámetros
best_rf_params = grid_search_rf.best_params_
print(f"Mejores parámetros para Random Forest: {best_rf_params}")

# Modelo ajustado con los mejores parámetros
random_forest_best = RandomForestRegressor(**best_rf_params, random_state=42)
random_forest_best.fit(X_train_reg_scaled, y_train_reg)

# Evaluación del modelo ajustado
y_pred_rf_best = random_forest_best.predict(X_test_reg_scaled)
rmse_rf_best = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf_best))
r2_rf_best = r2_score(y_test_reg, y_pred_rf_best)

print(f"\nRandom Forest Regressor Ajustado - RMSE: {rmse_rf_best:.2f} horas")
print(f"Random Forest Regressor Ajustado - R²: {r2_rf_best:.2f}")
"""
# ------------------------------
# 5. Random Forest Classifier
# ------------------------------
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
rf_classifier.fit(X_train_c_scaled, y_train_c)
y_pred_c = rf_classifier.predict(X_test_c_scaled)

accuracy = accuracy_score(y_test_c, y_pred_c)
print(f"\nPrecisión del Random Forest Classifier: {accuracy:.2f}")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_c, y_pred_c))

class_names = df['Muelle'].cat.categories.astype(str)
print("\nReporte de Clasificación:")
print(classification_report(y_test_c, y_pred_c, target_names=class_names, zero_division=0))

# Guardar el modelo Random Forest Regressor
with open('random_forest_regressor.pkl', 'wb') as f:
    pickle.dump(random_forest_best, f)

# Guardar el modelo Random Forest Classifier
with open('random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

# Guardar los escaladores
with open('scaler_reg.pkl', 'wb') as f:
    pickle.dump(scaler_reg, f)

with open('scaler_clf.pkl', 'wb') as f:
    pickle.dump(scaler_clf, f)

# Guardar las características utilizadas en ambos modelos
with open('features_reg.pkl', 'wb') as f:
    pickle.dump(features_reg, f)

with open('features_clf.pkl', 'wb') as f:
    pickle.dump(features_clf, f)

# Guardar los nombres de las clases para el modelo clasificador
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names.tolist(), f)
"""