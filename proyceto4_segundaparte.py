import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

# Cargar el archivo limpio
data = pd.read_csv("C:/Users/Stephany/Desktop/Nueva carpeta (3)/PuertoManta_IA/fecha_zarpe_cleaned.csv")

# Convertir la columna "Muelle" a valores categóricos
muelle_dummies = pd.get_dummies(data['Muelle'], prefix='Muelle')
data = pd.concat([data, muelle_dummies], axis=1)

# Crear etiquetas de clasificación para "Horas_Tardias"
def categorize_hours(hours):
    if hours <= 2:
        return 'baja'
    elif 2 < hours <= 5:
        return 'media'
    else:
        return 'alta'

data['Categoria_Estadia'] = data['Horas_Tardias'].apply(categorize_hours)

# Variables predictoras y objetivo
X = data[['Horas_Tardias'] + list(muelle_dummies.columns)]
y = data['Categoria_Estadia']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el árbol de decisión
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))

# Exportar la estructura del árbol
tree_rules = export_text(clf, feature_names=list(X.columns))
print("Reglas del Árbol de Decisión:\n", tree_rules)

# Guardar el resultado en un nuevo archivo CSV
output_path = '/mnt/data/fecha_zarpe_cleaned_with_categories.csv'
data.to_csv(output_path, index=False)
print(f"Archivo procesado guardado en: {output_path}")
