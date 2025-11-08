import pandas as pd
import random

class LocalizarHospital:
    """Implementación manual de K-Means sin numpy/math y lógica de generación condicional."""
    
    def __init__(self, K, M=100, N=100, num_casas=500, random_matrix_size=False, random_casas=False):
        self.K = K
        
        # Lógica Condicional para Tamaño de Matriz
        if random_matrix_size:
            self.M = random.randint(50, 200) # M y N aleatorios entre 50 y 200
            self.N = random.randint(50, 200)
        else:
            self.M = M
            self.N = N
            
        # Lógica Condicional para Vecindarios
        if random_casas:
            self.num_casas = random.randint(300, 1000) # Vecindarios aleatorios entre 300 y 1000
        else:
            self.num_casas = num_casas
            
        self.vecindarios = self._generar_vecindarios()
        self.hospitales = None
        self.visualizacion_posible = self.M <= 200 and self.N <= 200 # Usado en la API
        
        # Ajuste: Si el número de casas es menor que K, ajustamos K
        if len(self.vecindarios) < self.K:
            self.K = len(self.vecindarios)
            if self.K == 0:
                 raise ValueError("No se pudieron generar vecindarios únicos.")
            
    def _generar_vecindarios(self):
        """Genera coordenadas aleatorias para las casas, sin semilla fija."""
        x_coords = [random.randint(0, self.M - 1) for _ in range(self.num_casas)]
        y_coords = [random.randint(0, self.N - 1) for _ in range(self.num_casas)]
        
        casas = pd.DataFrame(list(zip(x_coords, y_coords)), columns=['x', 'y'])
        casas = casas.drop_duplicates().reset_index(drop=True)
        # Actualizamos el número de casas por si se eliminaron duplicados
        self.num_casas = len(casas) 
        return casas

    def _distancia_euclidiana_cuadrada(self, p1, p2):
        """Calcula la distancia euclidiana al cuadrado (Squared Euclidean Distance)."""
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def _ajustar_centroide_a_grilla(self, centroide, casas_df):
        """Ajusta un centroide a la posición de grilla vacía más cercana."""
        
        casas_set = set(map(tuple, casas_df[['x', 'y']].values))
        
        # Redondeo manual: (int(x + 0.5) si x >= 0)
        hospital_pos = (int(centroide[0] + 0.5) if centroide[0] >= 0 else int(centroide[0]),
                        int(centroide[1] + 0.5) if centroide[1] >= 0 else int(centroide[1]))
        
        if hospital_pos not in casas_set and 0 <= hospital_pos[0] < self.M and 0 <= hospital_pos[1] < self.N:
            return hospital_pos

        mejor_pos = None
        min_dist_sq = float('inf')
        
        # Buscamos la celda vacía más cercana dentro de los límites de la matriz
        for i in range(self.M):
            for j in range(self.N):
                celda = (i, j)
                if celda not in casas_set:
                    dist_sq = self._distancia_euclidiana_cuadrada(hospital_pos, celda)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        mejor_pos = celda
        
        return mejor_pos if mejor_pos else hospital_pos 

    def ubicar_hospitales(self, max_iter=100, tolerancia=0.001):
        """Implementación manual de K-Means."""
        
        X = self.vecindarios.values.tolist()
        
        # Si K se ajustó previamente a 0, detenemos.
        if self.K == 0:
            return 
        
        # 1. Inicialización: selección aleatoria de K puntos
        centroides = random.sample(X, self.K) 
        
        for _ in range(max_iter):
            # 2. Asignación (Clustering)
            clusters = [[] for _ in range(self.K)]
            for casa in X:
                distancias = [self._distancia_euclidiana_cuadrada(casa, c) for c in centroides]
                cluster_index = distancias.index(min(distancias))
                clusters[cluster_index].append(casa)
            
            # 3. Actualización (Recalcular Centroides)
            nuevos_centroides = []
            cambios_total_sq = 0
            for i, cluster in enumerate(clusters):
                if cluster:
                    df_cluster = pd.DataFrame(cluster, columns=['x', 'y'])
                    nuevo_centroide = (df_cluster['x'].mean(), df_cluster['y'].mean())
                    
                    cambios_total_sq += self._distancia_euclidiana_cuadrada(centroides[i], nuevo_centroide)
                    nuevos_centroides.append(nuevo_centroide)
                else:
                    # Mantiene el centroide anterior si el cluster está vacío
                    nuevos_centroides.append(centroides[i])

            centroides = nuevos_centroides
            if cambios_total_sq < tolerancia:
                break
                
        # 4. Ajuste final de las ubicaciones
        self.hospitales = [self._ajustar_centroide_a_grilla(c, self.vecindarios) 
                           for c in centroides]

    def obtener_resultados(self):
        """Devuelve los resultados finales, incluyendo los parámetros finales M, N y num_casas."""
        return {
            'K': self.K,
            'M': self.M,
            'N': self.N,
            'num_casas': self.num_casas,
            'vecindarios': self.vecindarios.to_dict(orient='records'),
            'hospitales': [{'x': p[0], 'y': p[1]} for p in self.hospitales] if self.hospitales else [],
            'visualizacion_posible': self.visualizacion_posible
        }