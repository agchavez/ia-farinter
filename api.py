# 1. Creación de la API con FastAPI
import pandas as pd
import torch
import joblib
import pyodbc  # Para conectarse a la base de datos
from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware
import traceback
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

# Configurar el registro de errores (logging)
logging.basicConfig(level=logging.INFO)

# Definir la estructura de los datos de entrada
class RequestData(BaseModel):
    suc_id: int
    dia_semana: int

# Crear la aplicación FastAPI
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir la arquitectura del modelo de red neuronal
class DotacionPersonalNN(torch.nn.Module):
    def __init__(self):
        super(DotacionPersonalNN, self).__init__()
        self.layer1 = torch.nn.Linear(4, 64)  # Capa de entrada (4 características)
        self.layer2 = torch.nn.Linear(64, 32)  # Capa oculta
        self.layer3 = torch.nn.Linear(32, 1)  # Capa de salida
        self.relu = torch.nn.ReLU()  # Función de activación ReLU

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Cargar el modelo y el scaler
model_path = 'model.pth'
scaler_path = 'scaler.pkl'
model = DotacionPersonalNN()
model.load_state_dict(torch.load(model_path))
model.eval()  # Cambiar a modo de evaluación
scaler = joblib.load(scaler_path)

# Cadena de conexión a la base de datos
connection_string = (
    'mssql+pyodbc://'
    'Angel_chavez:{}@172.16.2.227\\DWHFARINTERDEV/BI_FARINTER?'
    'driver=ODBC+Driver+17+for+SQL+Server'.format(quote_plus('@ng3l_ch@v3z'))
)

engine = create_engine(connection_string)


# Definir el endpoint para predecir la necesidad de personal
@app.post("/predict_by_history/")
def predict_staff(data: RequestData):
    try:
        logging.info(f"Datos recibidos en la solicitud: suc_id={data.suc_id}, dia_semana={data.dia_semana}")
        # Conectar a la base de datos y obtener los datos de transacciones promedio
        transacciones_por_hora = []
        try:
            with engine.connect() as conn:
                query = text("""
                SELECT
                    DATEPART(hour, fc.Factura_FechaHora) AS Hora,
                    COUNT(*) AS Transacciones_Totales
                FROM
                    [dbo].[BI_Kielsa_Hecho_FacturaEncabezado] AS fc
                WHERE
                    fc.AnioMes_Id = 202410 
                    AND fc.Emp_Id = 1
                    AND fc.Suc_Id = :suc_id
                    AND DATEPART(dw, fc.Factura_Fecha) = :dia_semana
                GROUP BY
                    DATEPART(hour, fc.Factura_FechaHora)
                ORDER BY
                    Hora
                """)
                result = conn.execute(query, {'suc_id': data.suc_id, 'dia_semana': data.dia_semana})
                logging.info("result: ", result)
                for row in result:
                    transacciones_por_hora.append(row)
        except Exception as db_e:
            logging.error(f"Error en la conexión a la base de datos o la consulta: {db_e}")
            logging.error(traceback.format_exc())
            return {"error": "Error al conectarse a la base de datos. Revisa los registros para más detalles."}

        # Verificar si hay datos de transacciones
        if len(transacciones_por_hora) == 0:
            return {"error": "No se encontraron transacciones para la combinación de sucursal y día de la semana proporcionados."}
        else:
            logging.info(f"Transacciones por hora: {transacciones_por_hora}")

        predicciones = []
        
        # Recorrer las 24 horas del día para calcular la necesidad de personal por cada hora
        for hora, transacciones_promedio in transacciones_por_hora:
            # Crear el DataFrame con los valores que tenemos
            input_data = pd.DataFrame({
                'Suc_Id': [data.suc_id],
                'Dia_Semana': [data.dia_semana],
                'Hora': [hora],
                'Transacciones_Totales': [transacciones_promedio]
            })

            # Normalizar los datos de entrada
            input_data_scaled = scaler.transform(input_data)
            
            # Convertir a tensor y ajustar la forma
            input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).view(1, -1)
            
            # Realizar la predicción
            with torch.no_grad():
                prediction = model(input_tensor)
                predicciones.append({"hora": hora, "personal_necesario": prediction.item(), "transacciones_promedio": transacciones_promedio})
                
        # ORdenar las predicciones por la hora del día
        predicciones = sorted(predicciones, key=lambda x: x["hora"])
        
        return predicciones
    except Exception as e:
        logging.error(f"Error al hacer la predicción: {e}")
        logging.error(traceback.format_exc())
        return {"error": "Ocurrió un error durante la predicción. Revisa los registros para más detalles."}


# API para predecir recibiendo todos los datos de entrada
@app.post("/predict_all/")
async def predict_all(request: Request, data: dict):
    try:
        logging.info(f"Petición recibida: {await request.body()}")
        input_data = pd.DataFrame({
            'Suc_Id': [data['suc_id']],
            'Dia_Semana': [data['dia_semana']],
            'Hora': [data['hora']],
            'Cantidad_Transacciones': [data['cantidad_transacciones']]
        })
        
        # Normalizar los datos de entrada
        input_data_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).view(1, -1)
        
        # Realizar la predicción
        with torch.no_grad():
            prediction = model(input_tensor)
            
        return {"personal_necesario": prediction.item()}
    
    except Exception as e:
        logging.error(f"Error al hacer la predicción: {e}")
        return {"error": "Ocurrió un error durante la predicción. Revisa los registros para más detalles."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000, reload=True)