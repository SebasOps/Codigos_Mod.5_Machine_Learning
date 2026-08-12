"""
API de predicción de calidad del agua.

Flujo:
1. Al arrancar, carga el pipeline (.pkl) UNA sola vez en memoria.
2. Expone /predict: recibe datos crudos del agua, devuelve la predicción del semáforo.
3. Expone /health: para revisar que la API está viva.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# ---------------------------------------------------------------
# 1. Cargar el pipeline una sola vez, cuando arranca la API
#    (no en cada request, sería muy lento)
# ---------------------------------------------------------------
pipeline = joblib.load("pipeline_calidad_agua.pkl")                    # TODO Nombre que ellos le den

app = FastAPI(title="API Calidad del Agua")


# ---------------------------------------------------------------
# 2. Definir la "forma" exacta de los datos que va a recibir /predict
#    Esto es un esquema Pydantic: valida tipos automáticamente.
# ---------------------------------------------------------------

                                                                        # TODO Cambiar por las que ellos usen
class DatosAgua(BaseModel):
    var1: float
    var2: float
    var3: float
    var4: float
    var5: float

    class Config: # Unicma
        json_schema_extra = {
            "example": {
                "var1": 7.2,
                "var2": 12.5,
                "var3": 6.8,
                "var4": 450.0,
                "var5": 22.0
            }
        }


# ---------------------------------------------------------------
# 3. Endpoint de salud: solo confirma que la API está corriendo
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------
# 4. Endpoint de predicción: el corazón de la API
# ---------------------------------------------------------------
@app.post("/predict")
def predict(datos: DatosAgua):
    try:
        # Pydantic ya validó tipos. Convertimos a DataFrame porque
        # el pipeline de sklearn espera esa forma (mismas columnas
        # y mismo orden con los que fue entrenado).
        entrada = pd.DataFrame([datos.model_dump()])

        # El pipeline hace todo internamente: escalado + predicción.
        # No hace falta transformar nada a mano aquí.
        prediccion = pipeline.predict(entrada)[0]

        # Si el modelo soporta predict_proba, devolvemos también
        # la confianza de la predicción (opcional pero útil para la interfaz)           TODO Revisar si lo hicieron
        probabilidades = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(entrada)[0]
            clases = pipeline.classes_
            probabilidades = {clase: round(float(p), 4) for clase, p in zip(clases, proba)}

        return {
            "semaforo": prediccion,
            "probabilidades": probabilidades                                           # TODO Revisar si lo hicieron
        }

    except Exception as e:
        # Si algo falla (dato raro, columna faltante, etc.) devolvemos
        # un error claro en vez de que la API se caiga sin explicación
        raise HTTPException(status_code=400, detail=str(e))
    