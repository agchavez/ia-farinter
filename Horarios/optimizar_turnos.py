import polars as pl
#import numpy as np
import random
import pulp
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

cfg = pl.Config(
    tbl_cols=-1,
    tbl_formatting="UTF8_FULL_CONDENSED",
    fmt_str_lengths=20,
    tbl_width_chars=2000,
    tbl_rows= 100,
    )


#30% ventas
#70% act. admin

def generar_dotacion_empleados(sucursales: list[str], 
                               H_1_5: list[int], 
                               H_6: list[int], 
                               H_7: list[int], 
                               seed: int = 42) -> pl.DataFrame:
    """
    Genera una tabla que representa la dotación de empleados necesaria por sucursal, semana, día y hora.

    Args:
        sucursales (list[str]): Lista de códigos de sucursales en formato de 3 caracteres.
        H_1_5 (list[int]): Lista de horas de operación para lunes a viernes.
        H_6 (list[int]): Lista de horas de operación para sábados.
        H_7 (list[int]): Lista de horas de operación para domingos.
        seed (int): Semilla para generar datos aleatorios (reproducibilidad).

    Returns:
        pl.DataFrame: Tabla de dotación de empleados por sucursal, semana, día y hora.
    """
    random.seed(seed)

    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    semanas = range(1, 5)  # Semanas 1 a 4
    dias = range(1, 8)  # Días de la semana 1 a 7

    # Generar registros base
    registros = []
    for sucursal in sucursales:
        for semana in semanas:
            for dia in dias:
                horas_operacion = H_1_5 if dia <= 5 else H_6 if dia == 6 else H_7
                for hora in horas_operacion:
                    necesidad = random.randint(1, 10)
                    registros.append({
                        "Sucursal_Numero": sucursal,
                        "Semana_Id": semana,
                        "Dia_Id": dia,
                        "Dia_Nombre": dias_semana[dia - 1],
                        "Hora_Id": hora,
                        "Necesidad_empleados": necesidad
                    })

    # Crear DataFrame inicial
    df = pl.DataFrame(registros)

    # Asegurar que "Necesidad_empleados" sea tipo Int32
    df = df.with_columns(pl.col("Necesidad_empleados").cast(pl.Int32))

    # Agrupar por sucursal, semana y día para calcular la media y desviación
    grupos = df.group_by(["Sucursal_Numero", "Semana_Id", "Dia_Id"]).agg(
        [
            pl.col("Necesidad_empleados").mean().alias("media"),
            pl.col("Necesidad_empleados").std().alias("desviacion")
        ]
    )

    # Calcular el coeficiente de variación (CV)
    grupos = grupos.with_columns(
        ((pl.col("desviacion") / pl.col("media")) * 100).alias("coef_variacion")
    )

    # Unir el CV al DataFrame original
    df = df.join(grupos, on=["Sucursal_Numero", "Semana_Id", "Dia_Id"], how="left")

    # Intercambiar registros donde el CV > 80% por la media móvil
    df = df.with_columns(
        pl.when(pl.col("coef_variacion") > 80)
        .then(pl.col("Necesidad_empleados").rolling_mean(window_size=3).fill_null("mean"))
        .otherwise(pl.col("Necesidad_empleados"))
        .alias("Necesidad_empleados")
    )
    df = df.with_columns([
    pl.col("Necesidad_empleados")
    .cast(pl.Float64)  # First cast to float
    .round(0)         # Round to nearest integer
    .cast(pl.Int64)   # Convert to integer
    .alias("Necesidad_empleados")
    ])

    # Eliminar columnas temporales
    df = df.drop(["media", "desviacion", "coef_variacion"])

    return df


# Ejemplo de uso
sucursales = ["001", "002", "121", "097"]
H_1_5 = list(range(6, 22))  # Horario de 6 AM a 8 PM de lunes a viernes
H_6 = list(range(7, 20))    # Horario de 7 AM a 8 PM para sábados
H_7 = list(range(7, 20))   # Horario de 7 AM a 8 PM para domingos
#table = generar_dotacion_empleados(sucursales, H_1_5, H_6, H_7)
#print(table)

def convertir_sets_a_listas(obj):
    """
    Convierte los conjuntos (`set`) en listas (`list`) dentro del diccionario
    para que sea serializable en JSON.
    """
    if isinstance(obj, dict):
        return {k: convertir_sets_a_listas(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_sets_a_listas(i) for i in obj]
    elif isinstance(obj, set):
        return list(obj)  # Convertir `set` a `list`
    else:
        return obj

def imprimir_reporte_json(reporte):
    """
    Imprime el diccionario `Reporte` en formato JSON con indentación.
    
    Parámetros:
    -----------
    reporte : dict
        Diccionario con la asignación de empleados a los turnos.
    """
    reporte_serializable = convertir_sets_a_listas(reporte)  # Convertir `set` a `list`
    print(json.dumps(reporte_serializable, indent=4, ensure_ascii=False))

def ajustar_horarios_bloque(Bloque_semana, Horarios):
    """
    Ajusta los horarios de apertura y cierre de Bloque_semana para coincidir con Horarios.

    Parámetros:
    -----------
    Bloque_semana : dict
        Diccionario con la demanda de personal por hora en cada fecha.
        { 'Sucursal': { 'Fecha': {Hora: Demanda, Hora: Demanda, ...} } }

    Horarios : dict
        Diccionario con los horarios de apertura y cierre de cada día de la semana.
        { "Lunes": (5.5, 20), "Martes": (6.5, 21), ... }

    Retorna:
    --------
    dict
        Bloque_semana ajustado con horarios corregidos y horas ordenadas.
    """

    # 🔹 Crear una copia para evitar modificar el diccionario original
    bloque_corregido = {}

    for sucursal, fechas in Bloque_semana.items():
        bloque_corregido[sucursal] = {}

        for fecha, demanda_horas in fechas.items():
            dia_semana = fecha.split()[1]  # Extraer "Lunes", "Martes", etc.
            h_apertura, h_cierre = Horarios[dia_semana]  # Obtener horario de la sucursal
            h_cierre = h_cierre - 1  #el inicio de la ultima hora de cierre.
            # 🔹 Obtener la primera y última hora en Bloque_semana
            horas_existentes = sorted(demanda_horas.keys())
            h_min = min(horas_existentes)
            h_max = max(horas_existentes)

            # 🔹 Nueva estructura con todas las horas corregidas
            nueva_fecha = dict(demanda_horas)  # Copia de los datos existentes

            # 🔹 Ajuste de apertura
            if h_apertura - h_min == 0.5:
                nueva_fecha.update({h_apertura: nueva_fecha[h_min]})  # Ajustar la primera hora
                del nueva_fecha[h_min]  # Eliminar la hora incorrecta
            elif h_apertura - h_min == -0.5:
                nueva_fecha.update({h_apertura: nueva_fecha[h_min]})  # Ajustar la primera hora
                #del nueva_fecha[h_min] 
            elif h_min > h_apertura:
                #print(f"la hora minima de demanda es {h_min} comenzo despues de la hora de apertura : {h_apertura}")
                # Agregar h_apertura como decimal si es necesario
                nueva_fecha.update({h_apertura: nueva_fecha[h_min]})
                #nueva_fecha.update({h_apertura: nueva_fecha[h_min]})
                # Agregar las horas enteras faltantes
                for h in range(int(h_apertura), int(h_min)):
                    nueva_fecha.update({h: nueva_fecha[h_min]})

            elif h_min < h_apertura:
                #print(f"la hora minima de demanda es {h_min} comenzo antes de la hora de apertura : {h_apertura}")
                # Eliminar todas las horas previas y poner h_apertura como nueva primera hora
                nueva_fecha = {h_apertura: nueva_fecha[h_min]}  # Se conserva la demanda original

            # 🔹 Ajuste de cierre
            if h_cierre - h_max == 0.5:
                nueva_fecha.update({h_cierre: nueva_fecha[h_max]})  # Ajustar la última hora
                #del nueva_fecha[h_max]  # Eliminar la hora incorrecta
            elif h_cierre - h_max == -0.5:
                nueva_fecha.update({h_cierre: nueva_fecha[h_max]})  # Ajustar la última hora
                del nueva_fecha[h_max]  # Eliminar la hora incorrecta
            #caso en el que la hora de cierre es mayor que la ultima hora de demanda
            elif h_max < h_cierre:
                # Agregar la hora de cierre como decimal si es necesario
                nueva_fecha.update({h_cierre: nueva_fecha[h_max]})
                # Agregar las horas enteras faltantes
                for h in range(int(h_max) + 1, int(h_cierre)):
                    nueva_fecha.update({h: nueva_fecha[h_max]})
            #caso en el que la ultima hora de demanda es mayor que la hora de cierra
            elif h_max > h_cierre:
                # Eliminar todas las horas posteriores a h_cierre y poner h_cierre como nueva última hora
                nueva_fecha.update({h_cierre: nueva_fecha[h_max]})  # Mantener la demanda original
                nueva_fecha = {h: nueva_fecha[h] for h in nueva_fecha if h <= h_cierre}
                

            # 🔹 Ordenar las horas de menor a mayor
            bloque_corregido[sucursal][fecha] = dict(sorted(nueva_fecha.items()))

    return bloque_corregido

def Inicios_turnos(Bloque_semana, tipos_turnos, fechas, sucursal):
    """
    Calcula los inicios de turnos para cada día de la semana, 
    excluyendo T6n si la sucursal cierra antes de las 22:00.
    """
    inicios_turno = {}
    #tol = 0.5 # Recoemndacion de Edwin, añadir 30 minutos de tolerancia para los turnos nocturnos y lograr que surtir demanda, si la sucursal cierra a las
    for fecha in fechas:
        # Obtener la última hora registrada en Bloque_semana (ya ajustada)
        hora_cierre_real = max(Bloque_semana[sucursal][fecha].keys()) 
        # Crear el diccionario de turnos para la fecha
        inicios_turno[fecha] = {}
        #print(f"hora_cierre_real: {hora_cierre_real}")
        for t in tipos_turnos:
            # 🔹 Si el turno es "T6n" y la sucursal cierra antes de las 22, no lo agregamos            
            if t == "T6n" and hora_cierre_real <= 21:
                continue  # Saltar este turno

            # Calcular las horas de inicio para este tipo de turno
            inicios_turno[fecha][t] = [
                h for h in Bloque_semana[sucursal][fecha].keys()
                if h + tipos_turnos[t]["duracion"] - 1 <= hora_cierre_real 
                and tipos_turnos[t]['rango_horas'][0] <= h
                and h + tipos_turnos[t]["duracion"] - 1 <= tipos_turnos[t]['rango_horas'][1]
                # or (t == 'T6n' and h + tipos_turnos[t]["duracion"] - 1 <= hora_cierre_real + tol
                # and tipos_turnos[t]['rango_horas'][0] <= h
                # and h + tipos_turnos[t]["duracion"] - 1 <= tipos_turnos[t]['rango_horas'][1] + tol)
            ]
            if tipos_turnos[t]['rango_horas'][1] >= hora_cierre_real and tipos_turnos[t]['rango_horas'][0] <= hora_cierre_real - tipos_turnos[t]["duracion"]:
                inicios_turno[fecha][t].append(hora_cierre_real - tipos_turnos[t]["duracion"] + 1)
                
            inicios_turno[fecha][t] = list(set(inicios_turno[fecha][t]))
            inicios_turno[fecha][t].sort()  

    return inicios_turno

def Generar_dict_sol(Bloque_semana, fechas, sucursal, Horarios):
    
    solucion = {
        sucursal: {
            fecha: {"T8d": [], "T7m": [], "T6n": []} for fecha in fechas
        }
    }

    # 🔹 Ajustar "Personal_necesario" y "Empleados_asignados"
    for fecha in fechas:
        dia_semana = fecha.split()[1]  # Extraer el día de la semana (Ej: "Lunes")
        apertura, cierre = Horarios[dia_semana]  # Obtener horario de la sucursal

        # 🔹 Filtrar la demanda eliminando horas antes de la apertura
        solucion[sucursal][fecha]["Personal_necesario"] = {
            h: Bloque_semana[sucursal][fecha][h]
            for h in Bloque_semana[sucursal][fecha] 
        }
        solucion[sucursal][fecha]["Empleados_asignados"] = {
            h: 0 for h in solucion[sucursal][fecha]["Personal_necesario"]
        }
    return solucion 

def crear_modelo(tipos_turnos, Bloque_semana, inicios_turno, fechas, sucursal):
    model = pulp.LpProblem(f"Asignacion_Turnos_Sucursal_{sucursal}", pulp.LpMinimize)

    # 🔹 Definir variables de decisión
    x = {}
    emp = {}
    subcov = {}
    overcov = {}

    # Variables de balance y cobertura
    # 🔹 Variables de empleados totales por hora (Solo en horas válidas según la apertura real)
    for fecha in fechas:
        # 🔹 Iterar sobre las horas ya corregidas en "Personal_necesario"
        #Horas = Bloque_semana[sucursal][fecha].keys()
        #Horas.append(inicios_turno[fecha][t])
        for h in Bloque_semana[sucursal][fecha].keys():
            # 🔹 Definir las variables de empleados, subcobertura y sobrecobertura para cada hora
            emp[(fecha, h)] = pulp.LpVariable(f"emp_{sucursal}_{fecha}_{h}", lowBound=0, cat=pulp.LpInteger)
            subcov[(fecha, h)] = pulp.LpVariable(f"sub_{sucursal}_{fecha}_{h}", lowBound=0, cat=pulp.LpInteger)
            overcov[(fecha, h)] = pulp.LpVariable(f"over_{sucursal}_{fecha}_{h}", lowBound=0, cat=pulp.LpInteger)

    # 🔹 Variables de asignación de turnos
    # Variables de asignación de turnos
    for fecha in fechas:
        # Obtener la última hora registrada en el Bloque_semana (ya ajustada)
        hora_cierre_real = max(Bloque_semana[sucursal][fecha].keys())

        for t in tipos_turnos:
            # 🔹 Si el tipo de turno es 'T6n', verificar si la hora de cierre es >= 22
            if t == "T6n" and hora_cierre_real < 21 :
                continue  # Si la sucursal cierra antes de las 22, no se define este turno desde las variable

            for h in inicios_turno[fecha][t]:
                # Calcular el upperBound para x[fecha,t,h]
                max_necesidad = max(
                    Bloque_semana[sucursal][fecha].get(h + i, 0) 
                    for i in range(tipos_turnos[t]["duracion"]) 
                    if (h + i) <= max(Bloque_semana[sucursal][fecha].keys())
                )
                upper_bound = max(10, max_necesidad + 5)
                
                # Definir la variable de decisión solo si pasa los filtros
                x[(fecha, t, h)] = pulp.LpVariable(f"x_{sucursal}_{fecha}_{t}_{h}", lowBound=0, upBound=upper_bound, cat=pulp.LpInteger)
    return model, x, emp, subcov, overcov
 
def asignacion_regentes(Bloque_semana, solucion, num_regentes):
    """
    Asigna los turnos fijos de los regentes y actualiza Bloque_semana y solucion.
    
    Parámetros:
    -----------
    Bloque_semana : dict
        Diccionario con la demanda de personal por hora en cada fecha.
    solucion : dict
        Diccionario donde se guardará la asignación de turnos de los regentes.
    num_regentes : int
        Número de regentes a asignar (máximo 2).
    """

    # 🔹 Limitar el número de regentes a 2
    if num_regentes > 2:
        print("⚠ ADVERTENCIA: Solo se pueden asignar 2 regentes por sucursal. Se asignarán solo 2.")
        num_regentes = 2

    sucursal = list(Bloque_semana.keys())[0]
    fechas = list(Bloque_semana[sucursal].keys())

    # 🔹 Calcular el promedio de demanda de empleados por día
    demanda_promedio = {
        fecha: sum(Bloque_semana[sucursal][fecha].values()) / len(Bloque_semana[sucursal][fecha])
        for fecha in fechas
    }
    # 🔹 Ordenar las fechas de mayor a menor demanda promedio
    fechas_ordenadas = sorted(demanda_promedio, key=demanda_promedio.get, reverse=True)

    # 🔹 Seleccionar los 6 días con mayor demanda
    fechas_seleccionadas = fechas_ordenadas[:6]

    # 🔹 Asignación del primer regente
    for i, fecha in enumerate(fechas_seleccionadas):
        h_apertura = min(Bloque_semana[sucursal][fecha].keys())  # Hora de apertura
        h_cierre = max(Bloque_semana[sucursal][fecha].keys())  # Hora de cierre

        # 🔹 Determinar el tipo de turno (8 horas o 4 horas)
        if i < 5:  # Primeros 5 días: Turnos de 8 horas
            tipo_turno = "T8d"
            duracion = 8
            h_inicio = h_apertura
        else:  # Último día: Turno de 4 horas
            tipo_turno = "T4"
            duracion = 4
            h_inicio = h_apertura

        # 🔹 Agregar el turno al diccionario de solución
        if tipo_turno not in solucion[sucursal][fecha]:  
            solucion[sucursal][fecha][tipo_turno] = []  # Crear la lista si no existe

        solucion[sucursal][fecha][tipo_turno].append({
            "Hora_entrada": h_inicio,
            "Hora_salida": h_inicio + duracion,
            "empleados": 1,
            "empleado": "Regente_1"
        })

        # 🔹 Si es un turno de 4 horas, mover "T4" al inicio del diccionario
        if tipo_turno == "T4":
            solucion[sucursal][fecha] = dict(
                [("T4", solucion[sucursal][fecha]["T4"])] +
                [(k, v) for k, v in solucion[sucursal][fecha].items() if k != "T4"]
            )

        # 🔹 Restar demanda en horas del turno
        for h in [h for h in Bloque_semana[sucursal][fecha].keys() if h_inicio <= h <= h_inicio + duracion]:
            if Bloque_semana[sucursal][fecha][h] >= 1:
                Bloque_semana[sucursal][fecha][h] -= 1  
            solucion[sucursal][fecha]["Empleados_asignados"][h] += 1

    # 🔹 Si hay un segundo regente, asignarlo
    if num_regentes == 2:
        # 🔹 Seleccionar los 6 días con mayor demanda para el segundo regente
        fechas_seleccionadas_2 = fechas_ordenadas[:6]

        # 🔹 Si los 6 días seleccionados son los mismos que el primer regente, reemplazar el último día
        if set(fechas_seleccionadas) == set(fechas_seleccionadas_2):
            fecha_sin_regente = next((fecha for fecha in fechas if fecha not in fechas_seleccionadas), None)
            if fecha_sin_regente:
                fechas_seleccionadas_2[-1] = fecha_sin_regente  # Reemplazar el último día con el día sin regente

        # 🔹 Asignación del segundo regente
        for i, fecha in enumerate(fechas_seleccionadas_2):
            h_apertura = min(Bloque_semana[sucursal][fecha].keys())  # Hora de apertura
            h_cierre = max(Bloque_semana[sucursal][fecha].keys())  # Hora de cierre

            # 🔹 Si el primer regente ya está asignado en esta fecha, el segundo entra a h_cierre - 7
            if any(turno["empleado"] == "Regente_1" for turno in solucion[sucursal][fecha].get("T8d", [])):
                h_inicio = h_cierre - 7
            else:
                h_inicio = h_apertura

            # 🔹 Determinar el tipo de turno
            if i < 5:
                tipo_turno = "T8d"
                duracion = 8
            else:
                tipo_turno = "T4"
                duracion = 4

            # 🔹 Agregar el turno del segundo regente
            if tipo_turno not in solucion[sucursal][fecha]:  
                solucion[sucursal][fecha][tipo_turno] = []

            solucion[sucursal][fecha][tipo_turno].append({
                "Hora_entrada": h_inicio,
                "Hora_salida": h_inicio + duracion,
                "empleados": 1,
                "empleado": "Regente_2"
            })

            # 🔹 Si es un turno de 4 horas, mover "T4" al inicio del diccionario
            if tipo_turno == "T4":
                solucion[sucursal][fecha] = dict(
                    [("T4", solucion[sucursal][fecha]["T4"])] +
                    [(k, v) for k, v in solucion[sucursal][fecha].items() if k != "T4"]
                )

            # 🔹 Restar demanda en horas del turno
            for h in [h for h in Bloque_semana[sucursal][fecha].keys() if h_inicio <= h <= h_inicio + duracion]:
                if Bloque_semana[sucursal][fecha][h] >= 1:
                    Bloque_semana[sucursal][fecha][h] -= 1  
                solucion[sucursal][fecha]["Empleados_asignados"][h] += 1
    #return Bloque_semana, solucion
  
def agregar_restricciones(Bloque_semana, model, x, emp, subcov, overcov, solucion, 
                          tipos_turnos, inicios_turno, fechas, sucursal, cost_sub, cost_over):
    
    #if num_regentes == 1:
        #cost_sub = 1.5
        #cost_over = 1
    # if num_regentes == 2:
    #     cost_sub = 1.25
    #     cost_over = 1
    # else:
    #cost_sub = 1.5
    #cost_over = 1
    #imprimir_reporte_json(Bloque_semana)
    model += (
        pulp.lpSum(
            tipos_turnos[t]['costo'] * x[(fecha, t, h)]
            for fecha in fechas
            for t in inicios_turno[fecha].keys()
            for h in inicios_turno[fecha][t]
            if (fecha, t, h) in x
        ) +
        pulp.lpSum(
            cost_sub * subcov[fecha, h] + cost_over * overcov[fecha, h]
            for fecha in fechas 
            for h in solucion[sucursal][fecha]["Personal_necesario"].keys()
        )
    ), "FuncionObjetivo"

    # 🔹 Restricción de balance general
    for fecha in fechas:
        horas = list(solucion[sucursal][fecha]['Personal_necesario'].keys())
        horas.sort()
        for h in horas:
            if (fecha, h) in emp:  # 🔹 Solo aplicar la restricción si emp[(fecha, h)] existe
                model += (
                    emp[(fecha, h)] + subcov[(fecha, h)] - overcov[(fecha, h)] == Bloque_semana[sucursal][fecha][h]
                ), f"Balance_Hora_{fecha}_{h}"

            
# 🔹 Restricción de empleados presentes en cada hora
    for fecha in fechas:
        horas = list(solucion[sucursal][fecha]['Personal_necesario'].keys())
        horas.sort()
        for h in solucion[sucursal][fecha]["Personal_necesario"].keys():
            model += (
                emp[(fecha, h)] == pulp.lpSum(
                    x[fecha, t, i]
                    for t in inicios_turno[fecha].keys()
                    for i in inicios_turno[fecha][t]
                    #for h_prime in solucion[sucursal][fecha]["Personal_necesario"].keys()
                    if i <= h <= i + tipos_turnos[t]["duracion"] -1
                )
             ), f"Definicion_Empleados_{fecha}_{h}"

     # 🔹 Restricción de al menos 1 empleado por hora solo si la demanda es >= 1
    for fecha in fechas:
        horas = list(solucion[sucursal][fecha]['Personal_necesario'].keys())
        horas.sort()
        for h in horas:
            if (fecha, h) in emp and Bloque_semana[sucursal][fecha][h] >= 1:
                model += (
                    emp[(fecha, h)] >= 1
                ), f"Minimo_1_Empleado_en_horas_con_demanda_{fecha}_{h}"


    # 🔹 Restricción de al menos un turno de tipo T8d (diurno) y T7m (mixto) por día
    #añadir verificacion de que la diferencia entre el inicio y el cierre sea mas de 8 horas
    # 🔹 Diccionario para registrar qué fechas ya tienen restricciones asignadas
    #restricciones_aplicadas = set()

    for fecha in fechas:
    #      # Si hay 8 horas o más y no se ha agregado una restricción en esta fecha
    #      # Replace the existing T8d constraint with this:
          if inicios_turno[fecha]["T8d"]:
    #          # Check if there are no T8d shifts already assigned in the solution for this date
              if not any(turno for turno in solucion[sucursal][fecha].get("T8d", [])):
                  model += (
                      pulp.lpSum(x[fecha, "T8d", h] for h in inicios_turno[fecha]["T8d"]) >= 1
                  ), f"Minimo_1_Turno_T8d_{fecha}"

            #restricciones_aplicadas.add(fecha)  # Registrar que ya se agregó una restricción para esta fecha

        # Si hay entre 7 y 8 horas y no se ha agregado una restricción en esta fecha
        # if inicios_turno[fecha]["T7m"] and fecha not in restricciones_aplicadas:
        #     model += (
        #         pulp.lpSum(x[fecha, "T7m", h] for h in inicios_turno[fecha]["T7m"]) >= 1
        #     ), f"Minimo_1_Turno_T7m_{fecha}"
        #     restricciones_aplicadas.add(fecha)  # Registrar que ya se agregó una restricción para esta fecha

                
        # # 🔹 Restricción de máximo 2 turnos nocturnos (T6n) por día
    # for fecha in fechas:
    #     model += (
    #         pulp.lpSum(x[fecha, "T6n", h] 
    #                    for h in solucion[sucursal][fecha]["Personal_necesario"].keys() 
    #                    if (fecha, "T6n", h) in x) <= 2
    #     ), f"Maximo_2_Turnos_T6n_{fecha}"

def solver_semana(Bloque_semana, Horarios, num_regentes, cost_sub, cost_over):
    """
    Resuelve la optimización de turnos semanales incluyendo vendedores,
    agrupando empleados en cada turno asignado.

    Parámetros:
    -----------
    Bloque_semana : dict
        Diccionario con la estructura:
        { 'Sucursal': { 'Fecha': {Hora: Demanda, Hora: Demanda, ...} } }
    
    Horarios : dict
        Diccionario con los horarios de apertura y cierre de cada día de la semana.
        { "Lunes": (6.5, 20), "Martes": (6.5, 20), ... }

    Retorna:
    --------
    dict
        Diccionario con la misma estructura que `Bloque_semana`, pero con los turnos asignados,
        agrupando empleados en cada turno.
    """
    sucursal = list(Bloque_semana.keys())[0]
    #Horarios = Obtener_Horario(sucursal)
    Bloque_semana = ajustar_horarios_bloque(Bloque_semana, Horarios)
    # Extraer la única sucursal
    
    fechas = list(Bloque_semana[sucursal].keys())

    # 🔹 Definir los tipos de turnos en intervalos de 0.5 horas
    #Cota_max_hora = max(Horarios.values(), key=lambda x: x[1])[1]
    
    tipos_turnos = {
        "T7m": {  
            'costo': 1, 
            "duracion": 7,
            "rango_horas": (10, 21.5),  # 14.0 a 22.5
        },
        "T8d": {  
            'costo': 1,
            "duracion": 8,
            "rango_horas": (5, 18),  # 5.0 a 18.0
        }, 
        "T6n": {  
            'costo': 1,
            "duracion": 6,
            "rango_horas": (17, 23),  # 17.0 a 23.0
        }
    }
    #print(f'tips_turnos: {tipos_turnos}')

    solucion = Generar_dict_sol(Bloque_semana, fechas, sucursal, Horarios) 
     # 🔹 Calcular los inicios de turnos para cada día de la seman
    inicios_turno = Inicios_turnos(Bloque_semana, tipos_turnos, fechas, sucursal)
    #imprimir_reporte_json(inicios_turno)
    #Limitar la suma
    #imprimir_reporte_json(Bloque_semana)
    if num_regentes > 0:
        #imprimir_reporte_json(Bloque_semana)
        asignacion_regentes(Bloque_semana, solucion, num_regentes)
        #imprimir_reporte_json(Bloque_semana)
    #imprimir_reporte_json(Bloque_semana)
    model, x, emp, subcov, overcov = crear_modelo(tipos_turnos, Bloque_semana, inicios_turno, fechas, sucursal)
    # 🔹 Restricciones
    agregar_restricciones(Bloque_semana, model, x, emp, subcov, overcov, solucion, 
                          tipos_turnos, inicios_turno, fechas, sucursal,
                          cost_sub, cost_over)
    
    
    # 🔹 Resolver el modelo
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    # 🔹 Obtener el estado de la solución
    status = pulp.LpStatus[model.status]
    costo_total = model.objective.value()  # Obtener el valor de la función objetivo

    # 🔹 Imprimir el estado de la solución y el costo total
    print(f"Estado de la solución para Sucursal {sucursal}: {status}")
    print(f"Costo total de la solución: {costo_total}")
    
    # 🔹 Extraer la solución de los vendedores 
    for (fecha, t, h), var in x.items():
        valor = pulp.value(var)  # Obtener el valor de la variable

        if valor is not None and int(valor) > 0:  # Verificar que no sea None antes de convertirlo a int
            solucion[sucursal][fecha][t].append({
                "Hora_entrada": h,
                "Hora_salida": h + tipos_turnos[t]["duracion"] ,
                "empleados": int(valor),
                "empleado": "Vendedores"
            })
    for (fecha, h), var in emp.items():
        valor = pulp.value(var)  # Obtener el valor de la variable
        if valor is not None :  # Verificar que no sea None antes de convertirlo a int
            solucion[sucursal][fecha]["Empleados_asignados"][h] +=  valor
            
    return solucion

def generar_horarios_sucursal(suc_id: str) -> pl.DataFrame:
    """
    Genera una tabla de horarios semanales para una sucursal.
    
    Args:
        suc_id (str): Identificador de la sucursal
        
    Returns:
        pl.DataFrame: Tabla con los horarios de la semana
    """
    # Generar datos
    registros = []
    random.seed(42)  # Para reproducibilidad
    
    for dia_id in range(1, 8):
        # Generar hora apertura (entre 5 y 10, puede ser .5)
        h_apertura = random.choice([x/2 for x in range(10, 21)])  # Genera 5, 5.5, 6, 6.5...10
        
        # Generar duración (entre 10 y 14 horas, puede ser .5)
        duracion = random.choice([x/2 for x in range(20, 29)])  # Genera 10, 10.5, 11...14
        
        h_cierre = h_apertura + duracion
        
        registros.append({
            "Suc_Id": suc_id,
            "Dia_Id": dia_id,
            "H_apertura": h_apertura,
            "H_cierre": h_cierre
        })
    
    # Crear DataFrame
    df = pl.DataFrame(registros)
    
    return df

def is_file_fresh(file_path, max_age_hours=1):
    path = Path(file_path)
    if not path.exists():
        return False
        
    file_time = datetime.fromtimestamp(path.stat().st_mtime)
    current_time = datetime.now()
    age = current_time - file_time
    
    return age < timedelta(hours=max_age_hours)

def cargar_tabla_horarios() -> pl.DataFrame:
    """
    Carga la tabla de horarios desde SQL.
    """
    query = """
    SELECT 
        Suc_Id,
        Dia_Id,
        H_apertura,
        H_cierre
    FROM DWH_Farinter.dbo.Dim_Horarios_Sucursales
    WHERE Estado = 1
    """
    
    # Conexión usando pyodbc
    conn_str = (
        "Driver={SQL Server};"
        "Server=SRVDWH;"
        "Database=DWH_Farinter;"
        "Trusted_Connection=yes;"
    )
    
    # Leer directamente a Polars DataFrame
    df = pl.read_database(query=query, connection_uri=conn_str)
    return df

def verificar(CACHE_PATH) -> pl.DataFrame:
    """
    Verifica si existe el archivo parquet en cache y lo carga.
    Si no existe, carga desde SQL y guarda en cache.
    """
    CACHE_PATH = Path(CACHE_PATH)
    if is_file_fresh(CACHE_PATH, max_age_hours=1):
        #print("Datos cargados desde cache")
        df = pl.read_parquet(CACHE_PATH)
    else:
        #print("Cache no encontrado, cargando desde la base de datos")
        df = cargar_tabla_horarios()
        
        # Crear directorio si no existe
        os.makedirs("cache", exist_ok=True)
        
        # Guardar en cache
        df.write_parquet(CACHE_PATH)
        
    return df

def Obtener_Horario(Suc_Id: str) -> dict:
    """
    Obtiene el horario de una sucursal específica desde la tabla en cache.
    
    Args:
        Suc_Id (str): Identificador de la sucursal
        
    Returns:
        dict: Diccionario con los horarios de la semana
    """
    # Mapeo de Dia_Id a nombres
    dias = {
        1: "Lunes",
        2: "Martes", 
        3: "Miércoles",
        4: "Jueves",
        5: "Viernes",
        6: "Sábado",
        7: "Domingo"
    }
    
    CACHE_PATH = "cache/horarios_sucursales.parquet"
    # Obtener tabla desde cache
    df = verificar(CACHE_PATH)
    
    # Filtrar por sucursal
    df_sucursal = df.filter(pl.col("Suc_Id") == Suc_Id)
    
    # Convertir a diccionario
    horarios = {}
    for row in df_sucursal.iter_rows():
        dia_nombre = dias[row[1]]
        horarios[dia_nombre] = (row[2], row[3])
    
    return horarios



Bloque_semana = {
    '001': {  # Código de la sucursal
        '03-02-2025 Lunes': {  # Fecha con el día de la semana
            5: 2, 6: 3, 7: 10, 8: 9, 9: 9, 10: 10, 11: 8, 12: 7, 13: 7, 14: 7, 15: 6, 16: 5, 17: 4, 18: 3, 19: 2, 20: 1, 21: 1, 22: 1
        },
        '04-02-2025 Martes': {
            6: 1, 7: 2, 8: 3, 9: 3, 10: 3, 11: 2, 12: 2, 13: 2, 14: 1, 15: 1, 16: 1, 17: 1, 18: 2, 19: 2, 20: 1, 21: 1
        },
        '05-02-2025 Miércoles': {
            5: 1, 6: 2, 7: 3, 8: 3, 9: 3, 10: 2, 11: 2, 12: 2, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 2, 19: 2, 20: 1
        },
        '06-02-2025 Jueves': {
            7: 2, 8: 3, 9: 3, 10: 3, 11: 2, 12: 2, 13: 1, 14: 1, 15: 1, 16: 1, 17: 2, 18: 2, 19: 2, 20: 1
        },
        '07-02-2025 Viernes': {
            6: 2, 7: 3, 8: 3, 9: 4, 10: 4, 11: 3, 12: 3, 13: 2, 14: 2, 15: 1, 16: 1, 17: 1, 18: 2, 19: 3, 20: 2
        },
        '08-02-2025 Sábado': {
            5: 1, 6: 2, 7: 2, 8: 3, 9: 3, 10: 2, 11: 2, 12: 2, 13: 1, 14: 1, 15: 1, 16: 1, 17: 2, 18: 2, 19: 1, 20: 1
        },
        '09-02-2025 Domingo': {
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 2, 15: 1, 16: 1, 17: 1, 18: 1, 19: 2, 20: 1
        }
    }
}

Bloque_semana_t1 = {
    '001': {  # Código de la sucursal
        '03-02-2025 Lunes': {  # Fecha con el día de la semana
            5: 2, 6: 2, 7: 2, 8: 2, 9: 1, 10: 1, 11: 3, 12: 3, 13: 3, 14: 2, 15: 2, 16: 2, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1
        },
        '04-02-2025 Martes': {
            6: 1, 7: 2, 8: 3, 9: 3, 10: 3, 11: 2, 12: 1, 13: 2, 14: 1, 15: 1, 16: 1, 17: 1, 18: 2, 19: 2, 20: 1, 21: 1
        },
        '05-02-2025 Miércoles': {
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1
        },
        '06-02-2025 Jueves': {
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1
        },
        '07-02-2025 Viernes': {
           6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1
        },
        '08-02-2025 Sábado': {
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1
        },
         '09-02-2025 Domingo': {
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1
        }
    }
}


#Entra hora de apertura y hora de cierre reales
Horarios = {
    "Lunes": (5.5, 22.5),
    "Martes": (6, 21),
    "Miércoles": (5.5, 20.5),
    "Jueves": (7.5, 20.5),
    "Viernes": (6, 20),
    "Sábado": (5.5, 20),
    "Domingo": (6.5, 20)
}
Horarios1 = {
    "Lunes": (5.5, 21),
    "Martes": (6, 21.5),
    "Miércoles": (6, 20.5),
    "Jueves": (6, 20.5),
    "Viernes": (6, 20),
    "Sábado": (6, 20),
    "Domingo": (6, 20)
}

sucursal = list(Bloque_semana.keys())[0]
fechas = list(Bloque_semana[sucursal].keys())

    # 🔹 Definir los tipos de turnos en intervalos de 0.5 horas
    #Cota_max_hora = max(Horarios.values(), key=lambda x: x[1])[1]
 
tipos_turnos = {
        "T7m": {  
            'costo': 1, 
            "duracion": 7,
            "rango_horas": (10, 21.5),  # 14.0 a 22.5
        },
        "T8d": {  
            'costo': 1,
            "duracion": 8,
            "rango_horas": (5, 18),  # 5.0 a 18.0
        }, 
        "T6n": {  
            'costo': 1,
            "duracion": 6,
            "rango_horas": (17,23),  # 17.0 a 23.0
        }
    }
    
num_regentes = 1
solution = solver_semana(Bloque_semana_t1, Horarios1, num_regentes,
                        cost_sub = 1.5, cost_over = 1)



def asignar_horarios_empleados(solucion, num_regentes):
    """
    Asigna los horarios semanales a los empleados minimizando el número de empleados necesarios.

    - Primero, extrae y asigna turnos de regentes en 'control'.
    - Luego, recorre los turnos de 'control' de mayor demanda a menor, asignándolos a empleados existentes o creando nuevos.
    - Si un turno de Vendedores tiene 'empleados' > 1, se asigna uno a uno hasta agotar ese número.
    - Cada empleado tiene 5 turnos pendientes al crearse; en cuanto se le asigna un turno, se reduce en 1.
    - Un empleado no puede asignarse más de 1 turno por día ni más de 6 turnos en la semana.

    Parámetros:
    -----------
    solucion : dict
        Diccionario con la información de turnos generada por solver_semana (incluye 'Vendedores' y 'Regente_i').
    num_regentes : int
        Número de regentes a asignar (0 si no hay regentes).

    Retorna:
    --------
    dict
        Diccionario con la asignación de empleados y sus horarios.
    """

    # Diccionario final de empleados
    empleados = {}
    id_empleado = 1       # Contador de empleados
    id_turno = 1          # Contador de ID de turnos

    sucursal = list(solucion.keys())[0]
    fechas = list(solucion[sucursal].keys())

    # Copiamos la solución para ir removiendo/ajustando turnos
    control = {}
    # Copia profunda
    from copy import deepcopy
    control[sucursal] = deepcopy(solucion[sucursal])

    # ----------------------------------------------------------------
    # 1. Asignar turnos de los regentes primero (si existen)
    # ----------------------------------------------------------------
    if num_regentes > 2:
        print("⚠ ADVERTENCIA: Solo se pueden asignar 2 regentes por sucursal. Se asignarán solo 2.")
        num_regentes = 2

    if num_regentes > 0:
        for i in range(1, num_regentes + 1):
            # Crear un empleado para este regente
            empleado_id = f"empleado_{id_empleado}"
            empleados[empleado_id] = {
                "Rol": f"Regente_{i}",
                "turnos_pendientes": 0,  # Los regentes ya tienen sus turnos fijos, no se calculan aquí
                "Horarios": []
            }

            # Recorrer fechas y tipos de turno
            for fecha in fechas:
                for tipo_turno, lista_turnos in control[sucursal][fecha].items():
                    # Solo T8d o T4 para regentes
                    if tipo_turno in ("T8d", "T4"):
                        # Buscar todos los turnos del regente i
                        regent_shifts = [t for t in lista_turnos if t["empleado"] == f"Regente_{i}"]
                        for shift in regent_shifts:
                            # Asignar este turno al empleado
                            empleados[empleado_id]["Horarios"].append({
                                "Dia": fecha,
                                "Turno": tipo_turno,
                                "Hora_entrada": shift["Hora_entrada"],
                                "Hora_salida": shift["Hora_salida"],
                                "Id_turno": id_turno
                            })
                            id_turno += 1
                        # Removerlos de 'control'
                        control[sucursal][fecha][tipo_turno] = [
                            x for x in lista_turnos if x["empleado"] != f"Regente_{i}"
                        ]

            id_empleado += 1

    # ----------------------------------------------------------------
    # 2. Ordenar las fechas por demanda promedio (mayor a menor)
    # ----------------------------------------------------------------
   # 🔹 Cambiar la forma de ordenar las fechas según la SUMA de turnos (empleados) en cada día
    turnos_por_fecha = {}
    for fecha in fechas:
        total_turnos = 0
        # Recorrer cada tipo de turno del día
        for tipo_turno, lista_turnos in control[sucursal][fecha].items():
            if isinstance(lista_turnos, list):  # Confirmar que sea una lista de turnos
                for turno in lista_turnos:
                    total_turnos += turno["empleados"]
        turnos_por_fecha[fecha] = total_turnos

    # 🔹 Ordenar de mayor a menor según la suma de 'empleados' (turnos) en ese día
    fechas_ordenadas = sorted(turnos_por_fecha, key=turnos_por_fecha.get, reverse=True)


    # ----------------------------------------------------------------
    # 3. Asignar los turnos a vendedores (control ya está sin regentes)
    # ----------------------------------------------------------------
    for fecha in fechas_ordenadas:
        # Recorremos T8d, T7m, T6n
        for tipo_turno in ["T8d", "T7m", "T6n"]:
            lista_turnos = control[sucursal][fecha].get(tipo_turno, [])
            # Recorremos cada turno en la lista
            for turno in lista_turnos:
                if turno["empleado"] == "Vendedores":
                    # Este turno puede tener 'empleados' > 1
                    num_req = turno["empleados"]  # Cantidad de empleados requeridos
                    hora_ent = turno["Hora_entrada"]
                    hora_sal = turno["Hora_salida"]
                    #count_emps = []
                    for _ in range(num_req):
                        asignado = False
                        count_mismo_tipo_dict = {}
                        count_emps = {}
                        mejor_empleado = None

                        for emp_id, datos in empleados.items():
                            # Filtrar por rol='Vendedor'
                            if not datos["Rol"].startswith("Vendedor"):
                                continue
                            if datos["turnos_pendientes"] <= 0:
                                continue
                            # Verificar que no tenga ya un turno este día
                            dias_trabajados = {h["Dia"] for h in datos["Horarios"]}
                            if fecha in dias_trabajados:
                                continue

                            # Para T6n, un empleado que ya tenga T8d/T7m no sirve
                            tipos_existentes = {h["Turno"] for h in datos["Horarios"]}
                            if tipo_turno == "T6n" and (("T8d" in tipos_existentes or "T7m" in tipos_existentes)):
                                continue

                            # Contar turnos del mismo tipo
                            count_mismo_tipo = sum(1 for h in datos["Horarios"] if h["Turno"] == tipo_turno)
                            if count_mismo_tipo > 0:
                                count_mismo_tipo_dict[emp_id] = count_mismo_tipo
                            
                            # Contar total de turnos
                            total_turnos = len(datos["Horarios"])
                            count_emps[emp_id] = total_turnos
                        
                        #for e in count_emps:
                        #    print(f'{e} tiene {count_emps[e]} turnos')
                        
                        # Asignar basado en count_mismo_tipo_dict
                        if count_mismo_tipo_dict:
                            # Obtener empleado con más turnos del mismo tipo
                            mejor_empleado = max(count_mismo_tipo_dict.items(), key=lambda x: x[1])[0]
                            datos = empleados[mejor_empleado]
                            datos["Horarios"].append({
                                "Dia": fecha,
                                "Turno": tipo_turno,
                                "Hora_entrada": hora_ent,
                                "Hora_salida": hora_sal,
                                "Id_turno": id_turno
                            })
                            datos["turnos_pendientes"] -= 1
                            id_turno += 1
                            asignado = True
                            #print(f'Se asignó el turno {fecha, tipo_turno, hora_ent}:, {mejor_empleado}')
                        
                        elif count_emps:
                            # Si no hay del mismo tipo, asignar al que tenga más turnos totales
                            mejor_empleado = max(count_emps.items(), key=lambda x: x[1])[0]


                        if not asignado and mejor_empleado is not None:
                            # Asignar el turno al 'mejor_empleado' con menos turnos pendientes
                            datos = empleados[mejor_empleado]
                            datos["Horarios"].append({
                                "Dia": fecha,
                                "Turno": tipo_turno,
                                "Hora_entrada": hora_ent,
                                "Hora_salida": hora_sal,
                                "Id_turno": id_turno
                            })
                            datos["turnos_pendientes"] -= 1
                            id_turno += 1
                            asignado = True

                        # 2) Si no se asignó a nadie, crear un nuevo empleado
                        if not asignado:
                            new_emp_id = f"empleado_{id_empleado}"
                            empleados[new_emp_id] = {
                                "Rol": "Vendedor",
                                "turnos_pendientes": 6,
                                "Horarios": []
                            }
                            empleados[new_emp_id]["Horarios"].append({
                                "Dia": fecha,
                                "Turno": tipo_turno,
                                "Hora_entrada": hora_ent,
                                "Hora_salida": hora_sal,
                                "Id_turno": id_turno
                            })
                            empleados[new_emp_id]["turnos_pendientes"] -= 1
                            id_turno += 1
                            id_empleado += 1

    # ----------------------------------------------------------------
    # 4. Asignar día libre a los empleados con 6 turnos
    # ----------------------------------------------------------------
    for emp_id, datos in empleados.items():
        if len(datos["Horarios"]) == 6:
            dias_ocupados = {h["Dia"] for h in datos["Horarios"]}
            dia_libre = next((f for f in fechas if f not in dias_ocupados), None)
            if dia_libre:
                datos["Dia_libre"] = dia_libre

    return empleados

#horarios_sucursal = Obtener_Horario("1")
#imprimir_reporte_json(horarios_sucursal)


Reporte = asignar_horarios_empleados(solution, num_regentes)
#imprimir_reporte_json(solution)
imprimir_reporte_json(Reporte)




Horarios = {
    'K001' :  {
    "Lunes": (7, 22),
    "Martes": (7, 22),
    "Miércoles": (7, 22),
    "Jueves": (7, 22),
    "Viernes": (7, 22),
    "Sábado": (7, 22),
    "Domingo": (7, 22)
    },

    'K002' :  {
        "Lunes": (7, 22),
        "Martes": (7, 22),
        "Miércoles": (7, 22),
        "Jueves": (7, 22),
        "Viernes": (7, 22),
        "Sábado": (7, 22),
        "Domingo": (7, 22)
    },

    'K004' :  {
        "Lunes": (7, 19),
        "Martes": (7, 19),
        "Miércoles": (7, 19),
        "Jueves": (7, 19),
        "Viernes": (7, 19),
        "Sábado": (7, 19),
        "Domingo": (7, 19)
    },

    'K005' :  {
        "Lunes": (7, 19),
        "Martes": (7, 19),
        "Miércoles": (7, 19),
        "Jueves": (7, 19),
        "Viernes": (7, 19),
        "Sábado": (7, 19),
        "Domingo": (7, 19)
    },

    'K006' :  {
        "Lunes": (8, 18),
        "Martes":  (8, 18),
        "Miércoles": (8, 18),
        "Jueves":  (8, 18),
        "Viernes": (8, 18),
        "Sábado": (8, 18),
        "Domingo":  (8, 18)
    },

    'K007' :  {
        "Lunes": (7.5, 21),
        "Martes":  (7.5, 21),
        "Miércoles": (7.5, 21),
        "Jueves":  (7.5, 21),
        "Viernes": (7.5, 21),
        "Sábado": (7.5, 21),
        "Domingo":  (7.5, 21)
    },
    
    'K008' :  {
        "Lunes": (7.5, 21),
        "Martes":  (7.5, 21),
        "Miércoles": (7.5, 21),
        "Jueves":  (7.5, 21),
        "Viernes": (7.5, 21),
        "Sábado": (7.5, 21),
        "Domingo":  (7.5, 21)
    },

    'K009' :  {
        "Lunes": (8, 18),
        "Martes":  (8, 18),
        "Miércoles": (8, 18),
        "Jueves":  (8, 18),
        "Viernes": (8, 18),
        "Sábado": (8, 18),
        "Domingo":  (9, 17)
    },

    'K010' :  {
        "Lunes": (8,20),
        "Martes":  (8, 20),
        "Miércoles": (8, 20),
        "Jueves":  (8, 20),
        "Viernes": (8, 20),
        "Sábado": (8, 21),
        "Domingo":  (8, 21)
    },
}

#imprimir_reporte_json(Horarios['K001'])