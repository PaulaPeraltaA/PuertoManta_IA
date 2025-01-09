import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Cargar el archivo CSV limpio
data = pd.read_csv("C:/Users/Stephany/Desktop/Nueva carpeta (3)/PuertoManta_IA/fecha_zarpe_cleaned.csv")

# Crear etiquetas de clasificación para "Horas_Tardias"
def categorize_hours(hours):
    if hours <= 2:
        return 'baja'
    elif 2 < hours <= 5:
        return 'media'
    else:
        return 'alta'

data['Categoria_Estadia'] = data['Horas_Tardias'].apply(categorize_hours)

# Convertir la columna "Muelle" a variables dummy
muelle_dummies = pd.get_dummies(data['Muelle'], prefix='Muelle', drop_first=True)
data = pd.concat([data, muelle_dummies], axis=1)

# Variables predictoras y objetivo
X = pd.concat([data[['Horas_Tardias']], muelle_dummies], axis=1)
y = data['Categoria_Estadia']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el árbol de decisión
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))

# Visualizar el árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True, fontsize=10)
plt.title("Árbol de Decisión para la Clasificación de Operaciones Portuarias")
plt.show()
