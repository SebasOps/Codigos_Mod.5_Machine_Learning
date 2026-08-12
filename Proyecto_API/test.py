"""
Este script SIMULA lo que harán los Científicos de Datos en su notebook.
Genera datos falsos, entrena un pipeline (preprocesamiento + modelo) y lo guarda en .pkl

TÚ NO NECESITAS ENTENDER ESTE ARCHIVO A FONDO — es solo para tener algo con qué
probar tu API mientras te entregan el archivo real. Cuando llegue el .pkl real,
este script ya no se usa para nada.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n = 300

# --- Datos falsos con las 5 variables genéricas ---
df = pd.DataFrame({
    "ph": np.random.uniform(5.5, 9.0, n),
    "turbidez": np.random.uniform(0, 50, n),
    "oxigeno_disuelto": np.random.uniform(2, 10, n),
    "conductividad": np.random.uniform(100, 1500, n),
    "temperatura": np.random.uniform(10, 30, n),
})

# --- Target falso (semáforo) basado en una regla simple ---
def semaforo(row):
    if row["ph"] < 6.5 or row["oxigeno_disuelto"] < 4:
        return "Rojo"
    elif row["turbidez"] > 25:
        return "Amarillo"
    else:
        return "Verde"

df["semaforo"] = df.apply(semaforo, axis=1)

X = df.drop(columns=["semaforo"])
y = df["semaforo"]

# --- Pipeline: preprocesamiento (escalado) + modelo, todo en un solo objeto ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "pipeline_calidad_agua.pkl")
print("Pipeline dummy guardado como 'pipeline_calidad_agua.pkl'")
print(f"Columnas que espera el modelo: {list(X.columns)}")
print(f"Clases posibles del target: {sorted(y.unique())}")