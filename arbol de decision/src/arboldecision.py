import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Preparación y Clasificación de Datos ---

def preparar_datos(df):
    """
    Prepara el DataFrame para el entrenamiento del modelo.
    Convierte las variables categóricas a numéricas.
    """
    # Renombrar columnas para usar nombres consistentes en el codigo
    df = df.rename(columns={
        'tiene_links': 'has_links',
        'errores_gramaticales': 'grammar_errors',
        'todo_mayusculas': 'all_caps',
        'solicita_info_personal': 'requests_personal_info',
        'etiqueta': 'label',
        'asunto': 'subject',
        'cuerpo': 'body',
        'remitente': 'sender'
    })

    df['label'].fillna(df['label'].mode()[0], inplace=True)
    df.dropna(subset=['label'], inplace=True)
    
    print(f"Total de correos utilizados para el proceso: {len(df)}")

    # Definir la lista de caracteristicas (features) que usara el modelo.
    features = [
        'feat_tiene_links', 'feat_errores_gramaticales', 'feat_todo_mayusculas', 
        'feat_solicita_info_personal', 'feat_cuerpo_len', 'feat_num_exclam', 
        'feat_upper_ratio', 'feat_num_words_body'
    ]

    # Convertir 'label' a numérico (spam=1, ham=0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Asignar las caracteristicas (X) y la variable de destino (y) para el entrenamiento.
    X = df[features]
    y = df['label']
    
    return X, y

def entrenar_y_evaluar(X, y, test_size, random_state):
    """
    Entrena un modelo de Árbol de Decisión y evalúa su desempeño.
    """
    # Dividir los datos en conjuntos de entrenamiento (para aprender) y prueba (para evaluar).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Mostrar el numero de correos usados en esta iteracion.
    print(f"  -> Iteracion {random_state + 1}:")
    print(f"     Conjunto de entrenamiento: {len(X_train)} correos")
    print(f"     Conjunto de prueba: {len(X_test)} correos")
    
    # Inicializar y entrenar el modelo de Arbol de Decisión (algoritmo CART por defecto).
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones sobre los datos de prueba.
    y_pred = modelo.predict(X_test)
    
    # Calcular metricas de desempeño: exactitud y F1-Score.
    exactitud = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # La funcion retorna los resultados de esta ejecucion.
    return exactitud, f1

def graficar_resultados(df_resultados, img_path):
    """
    Genera y guarda gráficos de caja (boxplots) para visualizar la distribución de las métricas.
    """
    # Crear una figura con dos subplots para los dos graficos de boxplot.
    plt.figure(figsize=(12, 6))
    
    # Primer subplot para Exactitud y F1-Score
    plt.subplot(1, 2, 1)
    plt.boxplot([df_resultados['Exactitud'], df_resultados['F1-Score']], labels=['Exactitud', 'F1-Score'])
    plt.title('Distribución de Métricas (50 Repeticiones)')
    plt.ylabel('Puntuación')
    
    # Segundo subplot para los Z-Scores de ambas métricas
    plt.subplot(1, 2, 2)
    plt.boxplot([df_resultados['Z-Score Exactitud'], df_resultados['Z-Score F1-Score']], labels=['Z-Score Exactitud', 'Z-Score F1-Score'])
    plt.title('Distribución de Z-Scores')
    plt.ylabel('Valor Z-Score')

    plt.tight_layout()
    
    # Guardar la grafica en la ruta especificada.
    grafico_path = os.path.join(img_path, 'metricas_boxplot.png')
    plt.savefig(grafico_path)
    print(f"\nGráfica de resultados guardada en:\n{grafico_path}")
    plt.show()

# --- Proceso Principal ---

def main():
    """
    Flujo completo: Clasificación y análisis de resultados usando un CSV existente.
    """
    # Se corrige la ruta para que sea relativa a la carpeta del script 'src'.
    # El '..' le dice a Python que suba un nivel de directorio para acceder a 'data'.
    csv_path = os.path.join('..', 'data', 'correos_limpios.csv')
    img_path = os.path.join('..', 'imgs')
    
    # Crear las carpetas de salida si no existen.
    os.makedirs(os.path.dirname(os.path.join('..', 'data', 'resultados_clasificacion.csv')), exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Error: El archivo '{csv_path}' no se encuentra.")
        print("Asegúrate de que la ruta relativa sea correcta y que el archivo exista.")
        return

    print(f"Cargando datos desde '{csv_path}'...")
    
    # Intentar cargar el archivo con multiples codificaciones para evitar errores
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"¡Archivo cargado exitosamente con la codificación '{encoding}'!")
            break
        except UnicodeDecodeError:
            print(f"Falló la codificación '{encoding}'. Intentando con la siguiente...")
        except Exception as e:
            print(f"Ocurrió un error inesperado al cargar el archivo: {e}")
            return
            
    if df is None:
        print("\nNo se pudo cargar el archivo con ninguna de las codificaciones probadas.")
        return

    # Mapear los nombres de las columnas en español a nombres en ingles para estandarizar.
    mapeo_columnas = {
        'asunto': 'subject', 'cuerpo': 'body', 'etiqueta': 'label', 'remitente': 'sender',
        'tiene_links': 'has_links', 'errores_gramaticales': 'grammar_errors',
        'todo_mayusculas': 'all_caps', 'solicita_info_personal': 'requests_personal_info',
        'clasificacion': 'classification'
    }
    df = df.rename(columns=mapeo_columnas)

    # Llamar a la funcion de preparacion de datos y obtener las variables X (features) y y (target).
    X, y = preparar_datos(df)

    # Iniciar el bucle para entrenar y evaluar el modelo 50 veces
    resultados = {'Exactitud': [], 'F1-Score': []}
    num_repeticiones = 50
    print(f"\nEjecutando la clasificación {num_repeticiones} veces...")

    for i in range(num_repeticiones):
        # En cada iteracion, entrenar y evaluar el modelo.
        exactitud, f1 = entrenar_y_evaluar(X, y, test_size=0.3, random_state=i)
        resultados['Exactitud'].append(exactitud)
        resultados['F1-Score'].append(f1)
    
    # Convertir los resultados de las 50 ejecuciones en un DataFrame.
    df_resultados = pd.DataFrame(resultados)
    
    # Calcular los Z-Scores para cada métrica.
    df_resultados['Z-Score Exactitud'] = zscore(df_resultados['Exactitud'])
    df_resultados['Z-Score F1-Score'] = zscore(df_resultados['F1-Score'])

    print("\n--- Resultados Promedio ---")
    print(df_resultados.describe())

    # Guardar los resultados en un archivo CSV en la ruta especificada.
    nombre_archivo_resultados = os.path.join('..', 'data', 'resultados_clasificacion.csv')
    df_resultados.to_csv(nombre_archivo_resultados, index=False)
    print(f"\nResultados guardados en '{nombre_archivo_resultados}'.")

    graficar_resultados(df_resultados, img_path)

if __name__ == "__main__":
    main()