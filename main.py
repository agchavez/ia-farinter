import pandas as pd
import torch
import joblib  # Para guardar y cargar el scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. Cargar el CSV y preprocesar los datos
def load_and_preprocess_data(csv_path):
    # Cargar el CSV
    df = pd.read_csv(csv_path)
    
    # Solo tomar la sucursal 1
    # df = df[df['Suc_Id'] == 1]
    
    # Seleccionar las características y la variable objetivo
    X = df[['Suc_Id', 'Dia_Semana', 'Hora', 'Transacciones_Totales']].values
    y = df['Personal_Necesario'].values

    # Normalizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convertir a tensores
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

# 2. Definir la red neuronal
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

# 3. Entrenar el modelo
def train_model(model, train_loader, optimizer, criterion, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Paso hacia adelante
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Paso hacia atrás y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 4. Evaluar el modelo y hacer predicciones
def evaluate_model(model, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred = model(X_test)

        # Mostrar las predicciones junto con los valores reales
        for true, pred in zip(y_test.cpu().numpy(), y_pred.cpu().numpy()):
            print(f"Real: {true[0]}, Predicción: {pred[0]}")

# 5. Crear un DataLoader para el entrenamiento
def create_dataloader(X_train, y_train, batch_size=4):
    train_data = TensorDataset(X_train, y_train)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 6. Guardar el modelo y el scaler
def save_model_and_scaler(model, scaler, model_path, scaler_path):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modelo guardado en {model_path}")
    print(f"Scaler guardado en {scaler_path}")

# 7. Función principal de entrenamiento
def main(csv_path):
    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(csv_path)

    # Crear el DataLoader
    train_loader = create_dataloader(X_train, y_train)

    # Inicializar el modelo
    model = DotacionPersonalNN()

    # Definir el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()  # Error cuadrático medio para regresión

    # Entrenar el modelo
    train_model(model, train_loader, optimizer, criterion, num_epochs=100)

    # Evaluar el modelo
    evaluate_model(model, X_test, y_test)
    
    # Guardar el modelo y el scaler
    save_model_and_scaler(model, scaler, 'model.pth', 'scaler.pkl')

"""
Funcion para cargar el modelo y hacer predicciones en base a datos de entrada
"""
# Función para cargar el modelo y hacer predicciones en base a datos de entrada
def predict(model, input_data, scaler):
    # Normalizar los datos de entrada
    input_data_scaled = scaler.transform(input_data)
    
    # Convertir a tensor
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    
    # Realizar la predicción
    with torch.no_grad():
        prediction = model(input_tensor)
        
    return prediction.item()

# Función para cargar los datos de entrada proporcionados por el usuario
def load_input_data():
    # Solicitar los datos al usuario
    suc_id = int(input("Ingrese el ID de la sucursal: "))
    dia_semana = int(input("Ingrese el día de la semana (1-7): "))
    hora = int(input("Ingrese la hora (0-23): "))
    cantidad_transacciones = int(input("Ingrese la cantidad de transacciones: "))
    cajas_activas = int(input("Ingrese la cantidad de cajas activas: "))
    dia_mes = int(input("Ingrese el día del mes: "))
    mes = int(input("Ingrese el mes (1-12): "))

    # Crear un diccionario con los datos de entrada
    input_data = {
        'Suc_Id': [suc_id],
        'Dia_Semana': [dia_semana],
        'Hora': [hora],
        'Cantidad_Transacciones': [cantidad_transacciones],
        'Cajas_Activas': [cajas_activas],
        'Dia_Mes': [dia_mes],
        'Mes': [mes]
    }
    
    # Crear un DataFrame con los datos de entrada
    input_df = pd.DataFrame(input_data)
    
    # Retornar el DataFrame
    return input_df

# Función principal para predecir el personal necesario
def main_predict(model_path, scaler_path):
    # Cargar el modelo
    model = DotacionPersonalNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Cambiar el modelo a modo de evaluación

    # Cargar los datos de entrada
    input_df = load_input_data()
    
    # Cargar el scaler previamente entrenado
    scaler = joblib.load(scaler_path)

    # Realizar la predicción
    prediction = predict(model, input_df, scaler)
    print(f"Se necesitarán {prediction:.2f} empleados para la sucursal.")
 
    
if __name__ == "__main__":
    # Ruta del CSV con los datos
    csv_path = 'personal_necesario.csv'  # Actualiza la ruta al archivo CSV
    main(csv_path)
    
    #main_predict('model.pth', 'scaler.pkl')
