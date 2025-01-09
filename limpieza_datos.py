import pandas as pd

# Cargar el archivo CSV
data = pd.read_csv("C:/Users/Stephany/Desktop/Nueva carpeta (3)/PuertoManta_IA/resultado_con_fecha_zarpe.csv")

# Eliminar las columnas "Carga Import." y "Carga Export."
data = data.drop(columns=['Carga Import.', 'Carga Export.'])

# Rellenar valores nulos en las columnas "Eslora" y "Horas_Tardias" con la media
data['Eslora'] = data['Eslora'].fillna(data['Eslora'].mean())
data['Horas_Tardias'] = data['Horas_Tardias'].fillna(data['Horas_Tardias'].mean())

# Guardar el resultado en un nuevo archivo CSV
output_path = 'fecha_zarpe_cleaned.csv'
data.to_csv(output_path, index=False)

print(f"Archivo procesado y guardado en: {output_path}")
