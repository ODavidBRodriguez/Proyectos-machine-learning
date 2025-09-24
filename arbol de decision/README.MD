# Proyecto de Clasificación de Spam con Árbol de Decisión

### Autor

* **Oscar David Barbosa Rodríguez** 
* **Introducción a Machine Learning - 801** 
* **Universidad de Cundinamarca** 

### Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar y evaluar un modelo de árbol de decisión para la clasificación de correos electrónicos como "spam" o "ham". Se utiliza un enfoque de aprendizaje automático supervisado para identificar patrones en las características de los correos, permitiendo al modelo tomar decisiones de clasificación precisas.

### Metodología

El proceso de clasificación se llevó a cabo siguiendo una serie de pasos:

1.  **Carga y Preprocesamiento de Datos:**
    * El dataset de entrada, `correos_limpios.csv`, fue cargado utilizando la biblioteca Pandas. Para asegurar la compatibilidad, se implementó un manejo robusto de codificaciones de caracteres (`UTF-8`, `latin-1`, etc.).
    * Se realizaron conversiones de datos, mapeando la columna `etiqueta` a valores numéricos (0 para "ham" y 1 para "spam"), un requisito para el entrenamiento del modelo.
    * Se eliminaron todas las filas que contenían valores nulos en la columna `etiqueta` para evitar errores durante el entrenamiento.

2.  **Entrenamiento y Evaluación del Modelo:**
    * Se utilizó el algoritmo Árbol de Decisión de la biblioteca Scikit-learn. Para evaluar la estabilidad y el rendimiento del modelo, el entrenamiento y la evaluación se repitieron 50 veces. En cada repetición, el dataset se dividió aleatoriamente en conjuntos de entrenamiento (70%) y prueba (30%).
    * Las métricas de desempeño utilizadas fueron la **Exactitud (Accuracy)** y el **F1-Score**, ya que son cruciales para un problema de clasificación de spam donde los falsos positivos y falsos negativos son importantes.

### Resultados

Los resultados de las 50 repeticiones fueron analizados para evaluar el rendimiento promedio y la consistencia del modelo.

* **Resultados Numéricos:**
    * Los valores de exactitud y F1-Score se guardaron en el archivo `resultados_clasificacion.csv` en la carpeta `data/`. Este archivo también incluye los Z-Scores, que indican la desviación de cada resultado con respecto a la media.
* **Análisis Visual:**
    * Se generó un gráfico de **boxplot** (`metricas_boxplot.png`) que muestra la distribución de las 50 métricas calculadas. El gráfico confirma que el modelo es **consistente**, con una variabilidad baja y un rendimiento que se mantiene cerca del promedio.

### Análisis y Conclusiones del Modelo

Los resultados del modelo de Árbol de Decisión demuestran una estabilidad y consistencia notables a lo largo de las 50 repeticiones de la evaluación. Esto indica que el modelo no solo es preciso, sino que su rendimiento es fiable, independientemente de la partición de datos utilizada para el entrenamiento y la prueba.

* Exactitud (Accuracy): La exactitud promedio del modelo se sitúa en un rango estrecho, con una mediana que ronda el 85%. La baja dispersión de los valores de exactitud, visible en el gráfico de boxplot, confirma que el modelo es consistentemente capaz de clasificar correctamente un alto porcentaje de correos electrónicos.

* F1-Score: El F1-Score muestra una mediana superior, cercana al 87.5%. Esta métrica es especialmente relevante en problemas de clasificación desbalanceada como el spam, ya que evalúa de manera equitativa la capacidad del modelo para identificar correctamente los correos spam (recall) y para evitar falsos positivos (precision). Un F1-Score alto y estable es un indicador sólido de la fiabilidad del modelo.

#### Explicación de las Variaciones

Las ligeras variaciones observadas en las métricas de desempeño son un comportamiento esperado. Estas fluctuaciones se deben a la división aleatoria de los datos en cada iteración. En cada ciclo, el conjunto de datos se divide aleatoriamente en subconjuntos de entrenamiento y prueba. A pesar de que la distribución de los datos es la misma, cada subconjunto puede variar ligeramente. Estas diferencias son la causa de las pequeñas variaciones en los resultados, aunque, como se observa en los gráficos, estas variaciones son mínimas.

### Ejecución del Proyecto

 **Ejecuta el Script, desde el bash:**
    
    cd "arbol de decision/src"
    python arboldecision.py

