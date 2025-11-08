import pandas as pd
import numpy as np


K_OPTIMO = 5
METRICA_OPTIMA = 'euclidiana'
# La columna de ID de usuario debe ser consistente
COLUMNA_ID_USUARIO = 'UserID_Limpio' 

class ModeloKNN:
    def __init__(self, calificaciones_df, canciones_df):
    
        self.calificaciones_df = self.preparar_dataframe(calificaciones_df.copy())
        self.canciones_df = canciones_df
        
        self.calificaciones_df[COLUMNA_ID_USUARIO] = self.calificaciones_df['UserID'].astype('Int64')
        
        self.matriz_usuarios_df = self.calificaciones_df.pivot_table(
            index=COLUMNA_ID_USUARIO, 
            columns='SongID', 
            values='Rating'  
        ).fillna(0)
        
        # Convertir a diccionario (para que el resto de tus funciones de KNN funcionen)
        self.matriz_usuarios = self.matriz_usuarios_df.T.to_dict()

        self.lista_ids_canciones = self.matriz_usuarios_df.columns.tolist() 
        self.promedios_usuarios = self.calcular_promedios_usuarios(self.matriz_usuarios)
        self.metadata = canciones_df.set_index('song_id') 
        
        self.limpiar_y_construir_datos()

    def preparar_dataframe(self, df):
        """Transforma el DataFrame de calificaciones de formato ancho a largo y estandariza nombres."""
        
        # Nombres de columna que DEBEN existir en el DataFrame final
        NOMBRES_FINALES = [
            'UserID', 
            'Edad', 
            'Genero', 
            'Region', 
            'GeneroFav', 
            'ClaseUsuario'
        ]

        # Obtener los nombres de las primeras 6 columnas (las de perfil)
        columnas_perfil_actuales = df.columns[:6].tolist()
        
        # Crear un diccionario para el renombre: el nombre actual se mapea al nombre final deseado
        mapeo_renombre = dict(zip(columnas_perfil_actuales, NOMBRES_FINALES))
        
        # Renombrar las columnas
        df.rename(columns=mapeo_renombre, inplace=True)

        # Las columnas de perfil (id_vars) son las que acabamos de renombrar
        id_vars_cols = NOMBRES_FINALES
        
        # Las columnas de calificación (value_vars) son todas las demás (las canciones 1, 2, 3...)
        value_vars_cols = [col for col in df.columns if col not in id_vars_cols]

        df_largo = df.melt(
            id_vars=id_vars_cols,
            value_vars=value_vars_cols,
            var_name='SongID',
            value_name='Rating'
        )

        # Convertir SongID y Rating a tipos numéricos (crucial para pivot_table)
        df_largo['SongID'] = pd.to_numeric(df_largo['SongID'], errors='coerce')
        df_largo['SongID'] = df_largo['SongID'].astype('Int64') # Int64 para soportar NaN si existiera

        # Convertir Rating a tipo float (es el valor)
        df_largo['Rating'] = pd.to_numeric(df_largo['Rating'], errors='coerce')

        # Convertir UserID a tipo entero (CRÍTICO para el index del pivot_table)
        df_largo['UserID'] = pd.to_numeric(df_largo['UserID'], errors='coerce').astype('Int64')


        # Eliminar filas donde no hay calificación (asumiendo que 0 es el valor de no calificado)
        df_largo = df_largo[df_largo['Rating'] > 0]
        df_largo.dropna(subset=['Rating', 'SongID', 'UserID'], inplace=True) 


                
        return df_largo

    def limpiar_y_construir_datos(self):
        """Prepara la matriz y calcula promedios iniciales."""
        
        columnas_canciones = [col for col in self.calificaciones_df.columns if str(col).isdigit()]
        self.lista_ids_canciones = [int(col) for col in columnas_canciones]
        
        for _, fila in self.calificaciones_df.iterrows():
            usuario_id = fila[COLUMNA_ID_USUARIO]
            self.matriz_usuarios[usuario_id] = {}
            for columna in columnas_canciones:
                calificacion = fila[columna]
                try:
                    calificacion_numerica = float(calificacion) if pd.notna(calificacion) else 0.0
                    self.matriz_usuarios[usuario_id][int(columna)] = calificacion_numerica
                except:
                    self.matriz_usuarios[usuario_id][int(columna)] = 0.0

        self.promedios_usuarios = self.calcular_promedios_usuarios(self.matriz_usuarios)

    def calcular_promedios_usuarios(self, matriz_usuarios):
        """Calcula el promedio de calificación para cada usuario."""
        promedios = {}
        for usuario_id, calificaciones in matriz_usuarios.items():
            calificaciones_validas = [calif for calif in calificaciones.values() if calif > 0]
            if calificaciones_validas:
                promedios[usuario_id] = sum(calificaciones_validas) / len(calificaciones_validas)
            else:
                promedios[usuario_id] = 0.0
        return promedios

    def calcular_similitud_euclidiana(self, usuario1_data, usuario2_data):
        """Calcula la distancia Euclidiana. Menor valor = mayor similitud."""
        distancia_total = 0.0
        canciones_comunes = 0
        
        for cancion_id in self.lista_ids_canciones:
            calif1 = usuario1_data.get(cancion_id, 0)
            calif2 = usuario2_data.get(cancion_id, 0)
            
            if calif1 > 0 and calif2 > 0:
                distancia_total += (calif1 - calif2) ** 2
                canciones_comunes += 1
        
        # Debe haber al menos 5 canciones en común (Mínimo requerido para una métrica robusta)
        if canciones_comunes < 5: 
            return float('inf') # Distancia infinita si no hay suficientes puntos
        
        return distancia_total ** 0.5

    def encontrar_vecinos_mas_cercanos(self, usuario_actual_id, k=K_OPTIMO, metrica=METRICA_OPTIMA):
        """Encuentra los k vecinos más cercanos al usuario actual."""
        usuario_actual_datos = self.matriz_usuarios.get(usuario_actual_id)
        if not usuario_actual_datos: return []
        
        distancias_usuarios = []
        
        for usuario_id, usuario_datos in self.matriz_usuarios.items():
            if usuario_id == usuario_actual_id:
                continue
            
            # Cálculo de la Distancia Euclidiana
            distancia = self.calcular_similitud_euclidiana(usuario_actual_datos, usuario_datos)
            distancias_usuarios.append((usuario_id, distancia))

        # Ordenar por distancia (menor valor primero)
        distancias_usuarios.sort(key=lambda x: x[1])
        
        # Filtrar distancias finitas y tomar los k primeros
        vecinos_finales = [v for v in distancias_usuarios if v[1] != float('inf')]
        
        return vecinos_finales[:k]
    
    def predecir_calificacion(self, usuario_actual_id, cancion_id, vecinos):
        """Predice la calificación de una canción usando la Desviación del Promedio Ponderada."""
        
        promedio_usuario_actual = self.promedios_usuarios.get(usuario_actual_id, 0.0)
        
        puntuacion_total_desviacion = 0.0
        peso_total_absoluto = 0.0
        
        for vecino_id, distancia in vecinos:
            calificacion_vecino = self.matriz_usuarios[vecino_id].get(cancion_id, 0)
            
            if calificacion_vecino > 0:
                # El peso es la inversa de la distancia (similitud)
                # W = 1 / (1 + Distancia)
                peso = 1.0 / (1.0 + distancia)
                
                # Desviación: calificación del vecino - promedio del vecino
                promedio_vecino = self.promedios_usuarios.get(vecino_id, 0.0)
                desviacion_vecino = calificacion_vecino - promedio_vecino
                
                puntuacion_total_desviacion += desviacion_vecino * peso
                peso_total_absoluto += abs(peso)
        
        if peso_total_absoluto > 0:
            prediccion = promedio_usuario_actual + (puntuacion_total_desviacion / peso_total_absoluto)
            # Limitar la predicción al rango [1.0, 5.0]
            return max(1.0, min(5.0, prediccion))
        else:
            return promedio_usuario_actual
        
    def generar_recomendaciones(self, usuario_actual_id, k=K_OPTIMO):
        """Genera las recomendaciones finales de canciones no calificadas."""
        
        vecinos = self.encontrar_vecinos_mas_cercanos(usuario_actual_id, k)
        if not vecinos:
            return []

        canciones_calificadas = set(self.matriz_usuarios.get(usuario_actual_id, {}).keys())
        
        candidatas = set()
        for vecino_id, _ in vecinos:
            candidatas.update(
                c_id for c_id, calif in self.matriz_usuarios[vecino_id].items() 
                if calif > 0 and c_id not in canciones_calificadas
            )

        recomendaciones_finales = []
        for cancion_id in candidatas:
            prediccion = self.predecir_calificacion(usuario_actual_id, cancion_id, vecinos)
            
            try:
                # Usamos el índice de la metadata limpia (SongID) para búsqueda O(1)
                meta = self.metadata.loc[cancion_id]
                
                # Adaptar los campos a los nombres internos usados:
                recomendaciones_finales.append({
                    "id": int(cancion_id),
                    "title": meta['Title'],      
                    "artist": meta['Artist'],    
                    "genre": meta['Genre'],      
                    "puntuacion_predicha": prediccion
                })
            except KeyError:
                # Ignorar canciones si el ID no se encuentra en la metadata
                continue
        
        recomendaciones_finales.sort(key=lambda x: x['puntuacion_predicha'], reverse=True)
        
        return recomendaciones_finales