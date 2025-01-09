import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Cargar el archivo CSV limpio
data = pd.read_csv("C:/Users/Stephany/Desktop/Nueva carpeta (3)/PuertoManta_IA/fecha_zarpe_cleaned.csv")

# Filtrar las columnas relevantes para el modelo
X = data[['Eslora', 'Muelle']]
y = data['Horas_Tardias']  # "Horas_Tardias" representa el tiempo de ocupación

# Convertir columnas categóricas a variables dummy (por ejemplo, "Muelle")
X = pd.get_dummies(X, columns=['Muelle'], drop_first=True)

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de regresión Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Realizar predicciones
predictions = ridge_model.predict(X_test)

# Evaluar el modelo
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Evaluación del modelo de regresión Ridge:")
print(f"RMSE (Root Mean Squared Error): {rmse}")
print(f"R² (Coeficiente de determinación): {r2}")

# Guardar el modelo entrenado
model_path = 'ridge_model.pkl'
joblib.dump(ridge_model, model_path)
print(f"Modelo entrenado guardado en: {model_path}")

# Crear interfaz gráfica para mostrar resultados
def mostrar_resultados():
    # Crear ventana principal
    ventana = tk.Tk()
    ventana.title("Visualización de Resultados del Modelo")
    ventana.geometry("600x400")

    # Mostrar coeficientes del modelo
    coeficientes = ridge_model.coef_
    caracteristicas = X.columns

    lbl_coef = tk.Label(ventana, text="Coeficientes del modelo:", font=("Arial", 12, "bold"))
    lbl_coef.pack(pady=10)

    tree = ttk.Treeview(ventana, columns=("Característica", "Coeficiente"), show="headings")
    tree.heading("Característica", text="Característica")
    tree.heading("Coeficiente", text="Coeficiente")
    
    for i, coef in enumerate(coeficientes):
        tree.insert("", "end", values=(caracteristicas[i], coef))
    tree.pack(pady=10)

    # Mostrar métricas de evaluación
    lbl_metricas = tk.Label(ventana, text="Métricas de Evaluación:", font=("Arial", 12, "bold"))
    lbl_metricas.pack(pady=10)

    lbl_rmse = tk.Label(ventana, text=f"RMSE: {rmse:.4f}", font=("Arial", 10))
    lbl_rmse.pack()

    lbl_r2 = tk.Label(ventana, text=f"R²: {r2:.4f}", font=("Arial", 10))
    lbl_r2.pack()

    # Botón para salir
    btn_salir = tk.Button(ventana, text="Salir", command=ventana.destroy, bg="red", fg="white", font=("Arial", 10, "bold"))
    btn_salir.pack(pady=20)

    # Iniciar bucle de la ventana
    ventana.mainloop()

# Llamar a la función para mostrar la interfaz gráfica
mostrar_resultados()
