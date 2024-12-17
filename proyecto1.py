import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Cargar datos
try:
    datos = pd.read_csv('resultado_con_fecha_zarpe.csv')

    # Verificar las columnas del archivo CSV
    print("Columnas disponibles:", datos.columns)

    # Seleccionar características y variables objetivo
    required_columns = ['Eslora', 'Muelle', 'Fecha_Llegada', 'ETD', 'Horas_Tardías', 'Retraso_Salida']
    
    # Verificar que todas las columnas requeridas estén presentes en el archivo CSV
    if all(col in datos.columns for col in required_columns):
        X = datos[['Eslora', 'Muelle', 'Fecha_Llegada', 'ETD']]  # Características
        y_regresion = datos['Horas_Tardías']  # Variable objetivo para regresión
        y_clasificacion = datos['Retraso_Salida']  # Variable objetivo para clasificación
    else:
        raise KeyError("Una o más columnas requeridas no están en el archivo CSV.")
except KeyError as e:
    print(f"Error de columna: {e}")
except FileNotFoundError as e:
    print(f"Error al cargar el archivo: {e}")

# Dividir los datos en conjuntos de entrenamiento y prueba
try:
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regresion, test_size=0.3, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clasificacion, test_size=0.3, random_state=42)

    # Normalizar características
    scaler = StandardScaler()
    X_train_reg = scaler.fit_transform(X_train_reg)
    X_test_reg = scaler.transform(X_test_reg)

    X_train_clf = scaler.fit_transform(X_train_clf)
    X_test_clf = scaler.transform(X_test_clf)

    # Convertir a tensores PyTorch
    X_train_reg = torch.tensor(X_train_reg, dtype=torch.float32)
    y_train_reg = torch.tensor(y_train_reg.values, dtype=torch.float32).view(-1, 1)
    X_test_reg = torch.tensor(X_test_reg, dtype=torch.float32)
    y_test_reg = torch.tensor(y_test_reg.values, dtype=torch.float32).view(-1, 1)

    X_train_clf = torch.tensor(X_train_clf, dtype=torch.float32)
    y_train_clf = torch.tensor(y_train_clf.values, dtype=torch.long)  # Para clasificación
    X_test_clf = torch.tensor(X_test_clf, dtype=torch.float32)
    y_test_clf = torch.tensor(y_test_clf.values, dtype=torch.long)
except KeyError as e:
    print(f"Error de división de datos: {e}")
except Exception as e:
    print(f"Error al preparar los datos: {e}")

# Cargar datos en DataLoaders
train_dataset_reg = TensorDataset(X_train_reg, y_train_reg)
test_dataset_reg = TensorDataset(X_test_reg, y_test_reg)

train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

train_dataset_clf = TensorDataset(X_train_clf, y_train_clf)
test_dataset_clf = TensorDataset(X_test_clf, y_test_clf)

train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

# Implementación del modelo de regresión lineal
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Una capa de red con una salida

    def forward(self, x):
        return self.fc(x)

# Crear modelo y definir la función de pérdida y el optimizador
model_reg = LinearRegressionModel(input_size=X_train_reg.shape[1])
criterion_reg = nn.MSELoss()  # Mean Squared Error Loss
optimizer_reg = optim.SGD(model_reg.parameters(), lr=0.01)

# Entrenamiento del modelo de regresión
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader_reg:
        optimizer_reg.zero_grad()
        outputs = model_reg(inputs)
        loss = criterion_reg(outputs, targets)
        loss.backward()
        optimizer_reg.step()

    # Mostrar el progreso cada 10 épocas
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluación del modelo de regresión
y_pred_reg = model_reg(X_test_reg)
rmse_reg = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
r2_reg = r2_score(y_test_reg, y_pred_reg)

print(f"RMSE (Regresión): {rmse_reg}")
print(f"R² (Regresión): {r2_reg}")

# Implementación del modelo de árbol de decisión
class DecisionTreeClassifierModel(nn.Module):
    def __init__(self, input_size):
        super(DecisionTreeClassifierModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Dos clases para clasificación binaria

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Crear modelo para clasificación
model_clf = DecisionTreeClassifierModel(input_size=X_train_clf.shape[1])
criterion_clf = nn.CrossEntropyLoss()  # Cross Entropy Loss para clasificación
optimizer_clf = optim.SGD(model_clf.parameters(), lr=0.01)

# Entrenamiento del modelo de clasificación
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader_clf:
        optimizer_clf.zero_grad()
        outputs = model_clf(inputs)
        loss = criterion_clf(outputs, targets)
        loss.backward()
        optimizer_clf.step()

    # Mostrar el progreso cada 10 épocas
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluación del modelo de clasificación
y_pred_clf = model_clf(X_test_clf).argmax(dim=1)
accuracy_clf = accuracy_score(y_test_clf, y_pred_clf)

print(f"Accuracy (Clasificación): {accuracy_clf}")

# Visualización del árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(model_clf, feature_names=['Eslora', 'Muelle'], filled=True)
plt.show()
