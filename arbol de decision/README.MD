# Proyecto de Clasificación de Spam con Árbol de Decisión

### Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar y evaluar un modelo de **Árbol de Decisión** para la clasificación de correos electrónicos como "spam" o "ham" (no spam). Se utiliza un enfoque de aprendizaje automático supervisado para identificar patrones en las características de los correos, permitiendo al modelo tomar decisiones de clasificación precisas.

### Metodología

El proceso de clasificación se llevó a cabo siguiendo una serie de pasos:

1.  **Carga y Preprocesamiento de Datos:**
    * El dataset de entrada, `correos_limpios.csv`, fue cargado utilizando la biblioteca **Pandas**. Para asegurar la compatibilidad, se implementó un manejo robusto de codificaciones de caracteres (`UTF-8`, `latin-1`, etc.).
    * Se realizaron conversiones de datos, mapeando la columna `etiqueta` a valores numéricos (0 para "ham" y 1 para "spam"), un requisito para el entrenamiento del modelo.
    * Se eliminaron todas las filas que contenían valores nulos en la columna `etiqueta` para evitar errores durante el entrenamiento.

2.  **Entrenamiento y Evaluación del Modelo:**
    * Se utilizó el algoritmo **Árbol de Decisión** de la biblioteca Scikit-learn. Para evaluar la estabilidad y el rendimiento del modelo de forma exhaustiva, el entrenamiento y la evaluación se repitieron 50 veces. En cada repetición, el dataset se dividió aleatoriamente en conjuntos de entrenamiento (70%) y prueba (30%).
    * Las métricas de desempeño utilizadas fueron la **Exactitud (Accuracy)** y el **F1-Score**, ya que son cruciales para un problema de clasificación de spam donde los falsos positivos y falsos negativos son importantes.

### Resultados

Los resultados de las 50 repeticiones fueron analizados para evaluar el rendimiento promedio y la consistencia del modelo.

* **Resultados Numéricos:**
    * Los valores de exactitud y F1-Score se guardaron en el archivo `resultados_clasificacion.csv` en la carpeta `data/`. Este archivo también incluye los Z-Scores, que indican la desviación de cada resultado con respecto a la media.
* **Análisis Visual:**
    * Se generó un gráfico de **boxplot** (`metricas_boxplot.png`) que muestra la distribución de las 50 métricas calculadas. El gráfico confirma que el modelo es **consistente**, con una variabilidad baja y un rendimiento que se mantiene cerca del promedio.

