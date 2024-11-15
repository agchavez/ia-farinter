import joblib
import numpy as np
import xgboost as xgb

# Nombres de las columnas según los datos de entrenamiento
columnas = ['clientes', 'turno', 'tipo_zona', 'ventas', 'eventos_locales']

def cargar_modelo():
    """Carga el modelo XGBoost desde un archivo .pkl."""
    try:
        modelo = joblib.load('modelo_xgboost_gpu.pkl')
        print("Modelo cargado exitosamente.")
        return modelo
    except FileNotFoundError:
        print("Error: No se encontró el archivo del modelo.")
        exit(1)

def predecir_personal(modelo, clientes, turno, tipo_zona, ventas, eventos_locales):
    """Realiza la predicción usando el modelo XGBoost."""
    try:
        # Preparar los datos en el formato adecuado
        datos_entrada = np.array([[clientes, turno, tipo_zona, ventas, eventos_locales]])
        dtest = xgb.DMatrix(datos_entrada, feature_names=columnas)

        # Realizar la predicción
        prediccion = modelo.predict(dtest)

        # Asegurar que la predicción sea un entero positivo
        return max(0, int(round(prediccion[0])))

    except Exception as e:
        print(f"Error durante la predicción: {str(e)}")
        exit(1)

def main():
    """Función principal que maneja la interacción con el usuario."""
    modelo = cargar_modelo()

    try:
        # Pedir datos al usuario
        clientes = int(input("Ingrese la cantidad de clientes esperados: "))
        turno = int(input("Ingrese el turno (1, 2, o 3): "))
        tipo_zona = int(input("Ingrese el tipo de zona (1 o 2): "))
        ventas = int(input("Ingrese las ventas proyectadas: "))
        eventos_locales = int(input("Ingrese el número de eventos locales (0-4): "))

        # Validar las entradas
        if turno not in [1, 2, 3] or tipo_zona not in [1, 2] or not (0 <= eventos_locales <= 4):
            raise ValueError("Turno, tipo de zona o número de eventos no válidos.")

        # Predecir la cantidad de personal
        cantidad_personal = predecir_personal(
            modelo, clientes, turno, tipo_zona, ventas, eventos_locales
        )
        print(f"Se recomienda una cantidad de {cantidad_personal} empleados para la proyección de ventas.")
    
    except ValueError as e:
        print(f"Entrada no válida: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
