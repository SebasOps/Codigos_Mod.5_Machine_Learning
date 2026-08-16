from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# ---------------------------------------------------------------
# 1. Cargar los 3 artefactos que generó el notebook
#    Orden exacto esperado por el pipeline
# ---------------------------------------------------------------
pipeline = joblib.load("pipeline_rf_semaforo.joblib")
label_encoder = joblib.load("label_encoder_semaforo.joblib")
columnas_features = joblib.load("columnas_features.joblib")  


# ---------------------------------------------------------------
# 2. Función de parseo — replica EXACTAMENTE la lógica del notebook
#    (celda 10, función limpiar_columna_medicion, factor_censura=0.5)
#    Vive aquí, a nivel de módulo, porque la usan varias mediciones
#    dentro del endpoint, no solo una.
# ---------------------------------------------------------------
def parsear_valor(texto: str) -> tuple[float, int]:
    """
    Recibe el texto tal cual lo escribe el usuario.
    - "7.2"     -> (7.2, 0)      valor medido normal, no censurado
    - "<0.005"  -> (0.0025, 1)   censurado: se usa límite/2, igual que en el notebook
    """
    texto = texto.strip()
    if texto.startswith("<"):
        limite = float(texto.lstrip("<").strip())
        return limite * 0.5, 1
    return float(texto), 0


app = FastAPI(title="API Calidad del Agua Subterránea")


# ---------------------------------------------------------------
# 3. Esquema de entrada — 13 columnas reales del modelo
#    Nota: nombres como "ALC_mg/L" no son identificadores válidos
#    de Python (por la barra), así que el atributo se llama distinto
#    (ej. alc_mgL) y usamos alias para que el JSON de afuera siga
#    usando el nombre real con barra.
# ---------------------------------------------------------------
class DatosAgua(BaseModel):
    alc_mgL: str = Field(alias="ALC_mg/L")
    conduct_mScm: str = Field(alias="CONDUCT_mS/cm")
    sdt_m_mgL: str = Field(alias="SDT_M_mg/L")
    fluoruros_mgL: str = Field(alias="FLUORUROS_mg/L")
    dur_mgL: str = Field(alias="DUR_mg/L")
    coli_fec_nmp: str = Field(alias="COLI_FEC_NMP/100_mL")
    n_no3_mgL: str = Field(alias="N_NO3_mg/L")
    as_tot_mgL: str = Field(alias="AS_TOT_mg/L")
    cr_tot_mgL: str = Field(alias="CR_TOT_mg/L")
    mn_tot_mgL: str = Field(alias="MN_TOT_mg/L")
    fe_tot_mgL: str = Field(alias="FE_TOT_mg/L")
 
    longitud: float = Field(alias="LONGITUD")
    latitud: float = Field(alias="LATITUD")
 
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "ALC_mg/L": "215.5",
                "CONDUCT_mS/cm": "815.0",
                "SDT_M_mg/L": "550.4",
                "FLUORUROS_mg/L": "0.5",
                "DUR_mg/L": "245.3",
                "COLI_FEC_NMP/100_mL": "<1.1",
                "N_NO3_mg/L": "2.08",
                "AS_TOT_mg/L": "<0.01",
                "CR_TOT_mg/L": "<0.004",
                "MN_TOT_mg/L": "0.15",
                "FE_TOT_mg/L": "0.35",
                "LONGITUD": -102.17,
                "LATITUD": 22.62
            }
        }
    }


# ---------------------------------------------------------------
# 4. Endpoints
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {"mensaje": "API de calidad del agua subterránea. Visita /docs para probarla."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "tipo_modelo": type(pipeline.named_steps["modelo"]).__name__,
        "clases": list(label_encoder.classes_),
        "n_features_esperadas": len(columnas_features),
        "columnas_features": columnas_features
    }


@app.post("/predict")
def predict(datos: DatosAgua):
    try:
        d = datos.model_dump(by_alias=True)
 
        # Mapeo: nombre real de la medición -> nombre real de su columna _censurado
        mediciones = [
            "ALC_mg/L", "CONDUCT_mS/cm", "SDT_M_mg/L", "FLUORUROS_mg/L", "DUR_mg/L",
            "COLI_FEC_NMP/100_mL", "N_NO3_mg/L", "AS_TOT_mg/L", "CR_TOT_mg/L",
            "MN_TOT_mg/L", "FE_TOT_mg/L"
        ]
 
        fila = {}
        for col in mediciones:
            valor_limpio, censurado = parsear_valor(d[col])
            fila[col] = valor_limpio
            fila[col + "_censurado"] = censurado
 
        fila["LONGITUD"] = d["LONGITUD"]
        fila["LATITUD"] = d["LATITUD"]
 
        # reordena exactamente como espera el pipeline (24 columnas ya completas)
        entrada = pd.DataFrame([fila])[columnas_features]
 
        pred_codificada = pipeline.predict(entrada)[0]
        semaforo = label_encoder.inverse_transform([pred_codificada])[0]
 
        probabilidades = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(entrada)[0]
            probabilidades = {
                clase: round(float(p), 4)
                for clase, p in zip(label_encoder.classes_, proba)
            }
 
        return {"semaforo": semaforo, "probabilidades": probabilidades}
 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Valor inválido: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    