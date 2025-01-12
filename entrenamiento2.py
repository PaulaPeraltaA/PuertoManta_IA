import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import pickle

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
if 'Horas_Tardias' in df.columns:
    df['Eslora_Horas_Tardias'] = df['Eslora'] * df['Horas_Tardias']
else:
    df['Eslora_Horas_Tardias'] = df['Eslora']

# Codificación de variables categóricas utilizando One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Agencia', 'Procedencia'], drop_first=True)

# ------------------------------
# 5. Definición de Variables para Modelos
# ------------------------------
# Variables para regresión
features_reg = [
    'Eslora', 'Dia_Semana', 'Mes', 'Hora_Llegada', 'Minuto_Llegada',
    'Eslora_Horas_Tardias', 'Horas_Tardias'
] + [col for col in df_encoded.columns if col.startswith('Procedencia_')]

X_reg = df_encoded[features_reg]
y_reg = df['Error_Horas']

# Variables para clasificación
features_clf = [
    'Eslora', 'Dia_Semana', 'Hora_Llegada', 'Mes', 'Eslora_Horas_Tardias'
] + [col for col in df_encoded.columns if col.startswith('Agencia_') or col.startswith('Procedencia_')]

X_clf = df_encoded[features_clf]
y_clf = df['Muelle'].cat.codes  # Usar codificación categórica para 'Muelle'

# ------------------------------
# 6. División de Datos en Entrenamiento y Prueba
# ------------------------------
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# ------------------------------
# 7. Escalado de Características
# ------------------------------
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

scaler_clf = StandardScaler()
X_train_c_scaled = scaler_clf.fit_transform(X_train_c)
X_test_c_scaled = scaler_clf.transform(X_test_c)

# ------------------------------
# 8. Regresión Ridge
# ------------------------------
param_grid = {'alpha': np.logspace(-4, 4, 50)}

grid_search = GridSearchCV(Ridge(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_reg_scaled, y_train_reg)

mejor_alpha = grid_search.best_params_['alpha']
modelo_ridge = Ridge(alpha=mejor_alpha, random_state=42)
modelo_ridge.fit(X_train_reg_scaled, y_train_reg)
y_pred_ridge = modelo_ridge.predict(X_test_reg_scaled)

rmse_ridge = np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge))
r2_ridge = r2_score(y_test_reg, y_pred_ridge)

print(f"\nRegresión Ridge - RMSE: {rmse_ridge:.2f} horas")
print(f"Regresión Ridge - R²: {r2_ridge:.2f}")

# ------------------------------
# 9. Random Forest Classifier
# ------------------------------
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid_rf,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_search_rf.fit(X_train_c_scaled, y_train_c)

best_rf_params = grid_search_rf.best_params_
rf_classifier_best = RandomForestClassifier(**best_rf_params, class_weight='balanced', random_state=42)
rf_classifier_best.fit(X_train_c_scaled, y_train_c)

y_pred_c_best = rf_classifier_best.predict(X_test_c_scaled)

accuracy_best = accuracy_score(y_test_c, y_pred_c_best)
print(f"\nPrecisión del Random Forest Classifier (Optimizado y Balanceado): {accuracy_best:.2f}")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_c, y_pred_c_best))

print("\nReporte de Clasificación:")
print(classification_report(y_test_c, y_pred_c_best, target_names=df['Muelle'].cat.categories.astype(str), zero_division=0))

joblib.dump(modelo_ridge, 'modelo_ridge.pkl')
# Guardar el modelo de Random Forest Classifier balanceado
joblib.dump(rf_classifier_best, 'modelo_rf_classifier.pkl')

# Guardar los escaladores
joblib.dump(scaler_reg, 'scaler_reg.pkl')
joblib.dump(scaler_clf, 'scaler_clf.pkl')

# Guardar las características utilizadas en los modelos
with open('features_reg.pkl', 'wb') as f:
    pickle.dump(features_reg, f)
 
with open('features_clf.pkl', 'wb') as f:
    pickle.dump(features_clf, f)

# Guardar las clases del modelo de clasificación
with open('class_names.pkl', 'wb') as f:
    pickle.dump(df['Muelle'].cat.categories.astype(str).tolist(), f)

