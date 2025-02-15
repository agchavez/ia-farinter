import os
from datetime import datetime, timedelta
from pathlib import Path
#import pyodbc
import polars as pl
import sqlalchemy
from sqlalchemy import create_engine
import math


# -----------------------------------------------------------------------------
# 1. Función que verifica la frescura de un archivo (is_file_fresh).
# -----------------------------------------------------------------------------
def is_file_fresh(file_path, max_age_hours=1):
    """
    Verifica si el archivo se creó o modificó hace menos de `max_age_hours`.
    Retorna True si el archivo existe y está dentro de la edad permitida,
    False en caso contrario.
    """
    path = Path(file_path)
    if not path.exists():
        return False

    file_time = datetime.fromtimestamp(path.stat().st_mtime)
    current_time = datetime.now()
    age = current_time - file_time
    return age < timedelta(hours=max_age_hours)


# -----------------------------------------------------------------------------
# 2. Clase para configuración y obtención de un Engine de SQLAlchemy
# -----------------------------------------------------------------------------
class SQLServerConfig:
    """
    Ajusta las credenciales, servidor y driver según tu entorno real.
    """
    def __init__(self, username, password, server, database, driver="ODBC Driver 17 for SQL Server"):
        self.username = username
        self.password = password
        self.server = server
        self.database = database
        self.driver = driver
        self.trust_server_certificate = "yes"  # Ajusta según tus necesidades

    def get_sqlalchemy_engine(self) -> sqlalchemy.engine.Engine:
        """
        Retorna un Engine para conectarse a SQL Server via pyodbc.
        El formato para 'mssql+pyodbc' es:
          mssql+pyodbc://user:pass@host/db?driver=Driver+Name
        """
        # Asegúrate de "escapar" los espacios en el driver con "+"
        # por ejemplo, "ODBC+Driver+17+for+SQL+Server"
        driver_escaped = self.driver.replace(" ", "+")
        
        # Construimos la cadena de conexión
        connection_str = (
            f"mssql+pyodbc://{self.username}:{self.password}"
            f"@{self.server}/{self.database}"
            f"?driver={driver_escaped}&TrustServerCertificate={self.trust_server_certificate}"
        )
        
        # Creamos el Engine
        engine = create_engine(connection_str, pool_pre_ping=True)
        return engine


# -----------------------------------------------------------------------------
# 3. Función para cargar la tabla de horarios usando Polars y un Engine.
# -----------------------------------------------------------------------------
def cargar_tabla_horarios(engine: sqlalchemy.engine.Engine) -> pl.DataFrame:
    """
    Ejecuta la consulta en SQL Server y retorna un DataFrame de Polars,
    usando 'connection=engine' en lugar de connection_uri.
    """
    query = """
    SELECT
        suc_id, 
        dia_id,
        h_apertura,
        h_cierre
    FROM [DL_FARINTER].[excel].[DL_Kielsa_Horario_Temp]
    """
    df = pl.read_database(
        query=query,
        connection=engine 
    )
    return df


# -----------------------------------------------------------------------------
# 4. Función principal que verifica el caché o carga la tabla y la guarda.
# -----------------------------------------------------------------------------
def verificar(CACHE_PATH: str, engine: sqlalchemy.engine.Engine) -> pl.DataFrame:
    """
    Verifica si existe el archivo parquet en caché y lo carga.
    Si no existe o está desactualizado (más de 1h), carga desde SQL y guarda en caché.
    Retorna un DataFrame de polars.
    """
    cache_path = Path(CACHE_PATH)

    if is_file_fresh(cache_path, max_age_hours=1):
        df = pl.read_parquet(cache_path)
    else:
        # Cargar desde la base de datos
        df = cargar_tabla_horarios(engine)
        # Crear el directorio si no existe
        os.makedirs(cache_path.parent, exist_ok=True)
        # Guardar en .parquet
        df.write_parquet(cache_path)

    return df


# -----------------------------------------------------------------------------
# 5. Instanciamos la configuración y creamos un Engine global (opcional).
#    En un proyecto real, manejar contraseñas de forma segura.
# -----------------------------------------------------------------------------
SQL_CFG = SQLServerConfig(
    username="Andrew_castejon",
    password="diez-veinte-diez",
    server="172.16.2.227\DWHFARINTERDEV",
    database="DL_FARINTER",
    driver="ODBC+Driver+17+for+SQL+Server"
)
#SQL_URL = SQL_CFG.get_sqlalchemy_url()
ENGINE = SQL_CFG.get_sqlalchemy_engine()
print(ENGINE)
# -----------------------------------------------------------------------------
# 6. Función para obtener los horarios de una sucursal (como dict).
# -----------------------------------------------------------------------------
def redondear(h_apertura, h_cierre) -> tuple:
    # Convert apertura to float and round to nearest .5
    h_ap = h_apertura.hour + h_apertura.minute/60
    h_ap = round(h_ap * 2) / 2  # Rounds to nearest 0.5
    
    # Convert cierre to float and handle special rounding
    h_ci = h_cierre.hour + h_cierre.minute/60
    
    # Round h_ci
    if h_ci % 1 >= 0.75:  # If minutes are 45 or more
        h_ci = math.ceil(h_ci)  # Round up to next hour
    else:
        h_ci = round(h_ci * 2) / 2  # Round to nearest 0.5
    
    return h_ap, h_ci

def Obtener_Horario(Suc_Id: str) -> dict:
    """
    Obtiene el horario de una sucursal específica desde el caché (o DB si no existe/está viejo).
    
    Args:
        Suc_Id (str): Identificador de la sucursal.
        
    Returns:
        dict: Diccionario con los horarios de la semana.
    """
    # Mapeo de Dia_Id a nombres
    dias = {
        1: "Lunes",
        2: "Martes",
        3: "Miercoles",
        4: "Jueves",
        5: "Viernes",
        6: "Sabado",
        7: "Domingo"
    }

    CACHE_PATH = "cache/horarios_sucursales.parquet"
    # Obtenemos la tabla de horarios (caché o DB)
    df = verificar(CACHE_PATH, ENGINE)

    # Filtrar por la sucursal deseada
    df_sucursal = df.filter(pl.col("suc_id") == Suc_Id)

    # Convertir a diccionario
    horarios = {}
    # row = (Suc_Id, Dia_Id, H_apertura, H_cierre)
    for row in df_sucursal.iter_rows():
        dia_id = row[1]
        dia_nombre = dias.get(dia_id, f"Desconocido_{dia_id}")
        h_apertura =  datetime.strptime(str(row[2]), '%H:%M:%S').time()
        h_cierre = datetime.strptime(str(row[3]), '%H:%M:%S').time()
        h_apertura, h_cierre = redondear(h_apertura, h_cierre)
        horarios[dia_nombre] = (h_apertura, h_cierre)
    return horarios



    

horarios = Obtener_Horario(int('107'))
print(horarios)
