from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import random


# Importar el nuevo módulo del modelo KNN
from modelo_knn import ModeloKNN, K_OPTIMO, METRICA_OPTIMA, COLUMNA_ID_USUARIO

app = Flask(__name__)
CORS(app)

# Cargar datasets reales
def load_datasets():
    """Carga los datasets de canciones y calificaciones usando try/except para las rutas."""
    
    # 1. Rutas para la METADATA DE CANCIONES (knn_model.metadata)
    rutas_metadata = [
        'modelo/metadata_canciones.csv',
        'backend/modelo/metadata_canciones.csv',
        '/app/modelo/metadata_canciones.csv',
        './modelo/metadata_canciones.csv'
    ]
    
    canciones_df = None
    
    for ruta_meta in rutas_metadata:
        try:
            canciones_df = pd.read_csv(ruta_meta)
            print(f"Metadata cargada desde: {ruta_meta}")
            break
        except Exception:
            # Continuar a la siguiente ruta si esta falla
            continue
    
    if canciones_df is None:
        print("ERROR: No se pudo cargar la metadata de canciones desde ninguna ruta.")
        return None, None
    
    # 2. Rutas para las CALIFICACIONES/MATRIZ (calificaciones_df)
    rutas_calificaciones = [
        'modelo/dataset_canciones.csv', 
        'backend/modelo/dataset_canciones.csv', 
        '/app/modelo/dataset_canciones.csv',
        './modelo/dataset_canciones.csv'
    ]
    
    calificaciones_df = None
    
    for ruta_cal in rutas_calificaciones:
        try:
            # Se mantiene 'latin-1' por si el archivo tiene caracteres especiales
            calificaciones_df = pd.read_csv(ruta_cal, encoding='latin-1') 
            print(f"Calificaciones cargadas desde: {ruta_cal}")
            break
        except Exception:
            # Continuar a la siguiente ruta si esta falla
            continue
        
    if calificaciones_df is None:
        print("No se pudo cargar el archivo de calificaciones desde ninguna ruta.")
        return None, None
    
    # 3. Reporte final
    try:
        print(f"Datasets cargados exitosamente.")
        return canciones_df, calificaciones_df
    except Exception as e:
        print(f"Error final al retornar DataFrames: {e}")
        return None, None

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "Recomendador Music Backend", "version": "1.0"})


# Cargar datos al iniciar
canciones_df, calificaciones_df = load_datasets()

# Inicializar el Modelo KNN si los datos fueron cargados
knn_model = None
if calificaciones_df is not None and canciones_df is not None:
    try:
        knn_model = ModeloKNN(calificaciones_df.copy(), canciones_df.copy())
        print("Modelo KNN inicializado correctamente.")
    except Exception as e:
        print(f"Error inicializando Modelo KNN: {e}")

# Crear lista de songs para la API
songs = []
if canciones_df is not None:
    for _, row in canciones_df.iterrows():
        songs.append({
            # ¡VERIFICADO con tus columnas!
            "id": row['id'],
            "title": row['nombre'],
            "artist": row['artista'],
            "genre": row['genero'],
            "year": row['año'],
            "popularity_base": row['popularidad_base']
        })
else:
    # Fallback a datos simulados si no se cargan los datasets
    songs = [
        {"id": 1, "title": "Shape of You", "artist": "Ed Sheeran", "genre": "Pop", "year": 2017, "popularity_base": 85},
        {"id": 2, "title": "Blinding Lights", "artist": "The Weeknd", "genre": "Pop", "year": 2019, "popularity_base": 90},
        {"id": 3, "title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "year": 1975, "popularity_base": 95},
        {"id": 4, "title": "Lose Yourself", "artist": "Eminem", "genre": "Rap", "year": 2002, "popularity_base": 88},
        {"id": 5, "title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Rock", "year": 1991, "popularity_base": 92},
        {"id": 6, "title": "Tusa", "artist": "Karol G", "genre": "Reggaeton", "year": 2019, "popularity_base": 95},
        {"id": 7, "title": "La Camisa Negra", "artist": "Juanes", "genre": "Rock Pop", "year": 2004, "popularity_base": 88},
        {"id": 8, "title": "Rebelión", "artist": "Joe Arroyo", "genre": "Salsa", "year": 1986, "popularity_base": 90},
    ]
    print("Usando datos de canciones simulados")

# Estructuras en memoria para sesiones activas
ratings = {}             # user_id -> { song_id: rating }
user_preferences = {}    # user_id -> { género: peso }

# Calcular popularidad basada en el dataset real (Se mantiene la lógica Content-Based para fallback)
def calcular_popularidad_real():
    """Calcula la popularidad de cada canción basada en las calificaciones del dataset"""
    if calificaciones_df is None:
        return {}
    
    popularidad = {}
    columnas_canciones = calificaciones_df.columns[6:]  
    
    for cancion_id in columnas_canciones:
        ratings_col = calificaciones_df[cancion_id]
        ratings_validas = ratings_col[ratings_col.between(1, 5)]
        
        if len(ratings_validas) > 0:
            avg_rating = ratings_validas.mean()
            num_ratings = len(ratings_validas)
            
            cancion_meta = canciones_df[canciones_df['id'] == int(cancion_id)]
            pop_base = cancion_meta['popularidad_base'].iloc[0] if not cancion_meta.empty else 50
            
            popularidad[cancion_id] = {
                'popularidad': (avg_rating * 15) + (num_ratings / 20) + (pop_base * 0.2),
                'avg_rating': avg_rating,
                'num_ratings': num_ratings
            }
        else:
            cancion_meta = canciones_df[canciones_df['id'] == int(cancion_id)]
            pop_base = cancion_meta['popularidad_base'].iloc[0] if not cancion_meta.empty else 50
            popularidad[cancion_id] = {
                'popularidad': pop_base,
                'avg_rating': 0,
                'num_ratings': 0
            }
    
    return popularidad

# Calcular popularidad al inicio
popularidad_real = calcular_popularidad_real() if calificaciones_df is not None else {}

def calcular_preferencias_genero(user_id):
    """Calcula pesos para cada género basado en las calificaciones del usuario"""
    user_ratings = ratings.get(user_id, {})
    
    if not user_ratings:
        return {"Pop": 1.0}
    
    genero_ratings = {}
    for song_id, rating in user_ratings.items():
        song = next((s for s in songs if s["id"] == song_id), None)
        if song:
            genero = song["genre"]
            if genero not in genero_ratings:
                genero_ratings[genero] = []
            genero_ratings[genero].append(rating)
    
    genero_pesos = {}
    total_peso = 0
    
    for genero, ratings_list in genero_ratings.items():
        avg_rating = sum(ratings_list) / len(ratings_list)
        num_ratings = len(ratings_list)
        
        peso = (avg_rating * num_ratings) ** 0.7
        genero_pesos[genero] = peso
        total_peso += peso
    
    if total_peso > 0:
        for genero in genero_pesos:
            genero_pesos[genero] = genero_pesos[genero] / total_peso
    
    generos_ordenados = dict(sorted(genero_pesos.items(), key=lambda x: x[1], reverse=True))
    
    print(f"Preferencias de género para {user_id}: {generos_ordenados}")
    return generos_ordenados


@app.route("/recommendations/<user_id>", methods=["GET"])
def recommendations(user_id):
    user_ratings = ratings.get(user_id, {})
    
    # Condición para usar KNN: al menos 5 calificaciones y el modelo debe existir
    if len(user_ratings) >= 5 and knn_model is not None:
        print(f"Usando KNN para {user_id} (ratings: {len(user_ratings)})")
        
        # El modelo KNN espera el user_id como el tipo de la columna UserID_Limpio.
        # En el caso de los usuarios de sesión, es un string.
        recomendaciones_knn = knn_model.generar_recomendaciones(user_id, k=K_OPTIMO)
        
        # Limpiar el formato de la recomendación (ya viene ordenado por predicción)
        recs_limpias = [{
            "id": r['id'],
            "title": r['title'],
            "artist": r['artist'],
            "genre": r['genre'],
            "pred_rating": f"{r['puntuacion_predicha']:.2f}" # Retornar la predicción para el front
        } for r in recomendaciones_knn[:8]]
        
        if recs_limpias:
            return jsonify(recs_limpias)
        
        print("KNN no encontró vecinos. Recurriendo a Content-Based.")

    # FALLBACK: Si es usuario nuevo o KNN falla, usar Content-Based (popularidad/género)
    if not user_ratings:
        # Recomendación por popularidad global (para nuevos)
        canciones_ordenadas = sorted(
            songs, 
            key=lambda x: popularidad_real.get(str(x["id"]), {}).get('popularidad', x["popularity_base"]), 
            reverse=True
        )
        return jsonify(canciones_ordenadas[:8])
    
    # Recomendación por género (Content-Based)
    generos_usuario = calcular_preferencias_genero(user_id)
    recs = obtener_recomendaciones_por_generos(user_id, generos_usuario)
    
    return jsonify(recs[:8])

# Se mantiene la función Content-Based como fallback
def obtener_recomendaciones_por_generos(user_id, generos_usuario):
    """Obtiene recomendaciones considerando múltiples géneros (Content-Based)"""
    user_rated = set(ratings.get(user_id, {}).keys())
    
    generos_principales = list(generos_usuario.keys())[:3]
    
    candidatos = []
    
    for genero in generos_principales:
        peso = generos_usuario[genero]
        
        canciones_genero = [
            s for s in songs 
            if s["genre"] == genero and s["id"] not in user_rated
        ]
        
        for cancion in canciones_genero[:15]:
            pop_score = popularidad_real.get(str(cancion["id"]), {}).get('popularidad', cancion["popularity_base"])
            score_final = pop_score * peso
            candidatos.append({
                "cancion": cancion,
                "score": score_final,
                "genero": genero,
                "peso_genero": peso
            })
    
    if len(candidatos) < 8:
        generos_restantes = [g for g in set(s["genre"] for s in songs) if g not in generos_principales]
        
        for genero in generos_restantes:
            canciones_genero = [
                s for s in songs 
                if s["genre"] == genero and s["id"] not in user_rated
            ]
            
            for cancion in canciones_genero[:5]:
                pop_score = popularidad_real.get(str(cancion["id"]), {}).get('popularidad', cancion["popularity_base"])
                score_final = pop_score * 0.1
                candidatos.append({
                    "cancion": cancion,
                    "score": score_final,
                    "genero": genero,
                    "peso_genero": 0.1
                })
    
    candidatos_ordenados = sorted(candidatos, key=lambda x: x["score"], reverse=True)
    
    canciones_vistas = set()
    recomendaciones_finales = []
    
    for cand in candidatos_ordenados:
        if cand["cancion"]["id"] not in canciones_vistas:
            canciones_vistas.add(cand["cancion"]["id"])
            recomendaciones_finales.append(cand["cancion"])
        
        if len(recomendaciones_finales) >= 15:
            break
    
    print(f"Recomendaciones (CB) para {user_id}: {[r['title'] for r in recomendaciones_finales[:8]]}")
    return recomendaciones_finales

#  Obtener canciones no calificadas 
@app.route("/songs_not_rated/<user_id>", methods=["GET"])
def songs_not_rated(user_id):
    user_rated = ratings.get(user_id, {})
    not_rated = [s for s in songs if s["id"] not in user_rated]
    
    canciones_mezcladas = not_rated.copy()
    random.shuffle(canciones_mezcladas)
    
    seleccion_aleatoria = canciones_mezcladas[:15]
    
    seleccion_ordenada = sorted(
        seleccion_aleatoria,
        key=lambda x: popularidad_real.get(str(x["id"]), {}).get('popularidad', x["popularity_base"]),
        reverse=True
    )
    
    print(f"Carrusel para {user_id}: {len(seleccion_ordenada)} canciones")
    print(f"Géneros en carrusel: {list(set(c['genre'] for c in seleccion_ordenada))}")
    
    return jsonify(seleccion_ordenada[:12])


# Calificar una canción 
@app.route("/rate_song", methods=["POST"])
def rate_song():
    data = request.json
    user_id = str(data.get("user_id"))
    song_id = int(data.get("song_id"))
    rating = int(data.get("rating"))

    if rating < 1 or rating > 5:
        return jsonify({"error": "La calificación debe estar entre 1 y 5."}), 400

    if user_id not in ratings:
        ratings[user_id] = {}

    # Guardar la calificación en la sesión (para Content-Based)
    ratings[user_id][song_id] = rating

    if knn_model is not None:
        # Añadir o actualizar el usuario en la matriz del modelo KNN
        if user_id not in knn_model.matriz_usuarios:
            knn_model.matriz_usuarios[user_id] = {}
        
        # Asegurar que la canción existe como columna en el modelo
        if song_id not in knn_model.lista_ids_canciones:
            knn_model.lista_ids_canciones.append(song_id)

        # Actualizar la calificación
        knn_model.matriz_usuarios[user_id][song_id] = float(rating)
        
        # Recalcular el promedio del usuario actual
        knn_model.promedios_usuarios = knn_model.calcular_promedios_usuarios(knn_model.matriz_usuarios)

    # Recalcular preferencias de género
    generos_usuario = calcular_preferencias_genero(user_id)
    user_preferences[user_id] = generos_usuario

    # Información de debug
    song = next((s for s in songs if s["id"] == song_id), None)
    genero_cancion = song["genre"] if song else "Desconocido"
    
    print(f"Usuario {user_id} calificó '{song['title'] if song else '?'}' ({genero_cancion}) con {rating} estrellas")
    print(f"Preferencias actualizadas: {generos_usuario}")

    return jsonify({
        "message": "Calificación guardada correctamente.",
        "genero_cancion": genero_cancion,
        "preferencias_actuales": generos_usuario
    })

# Clasificación de usuarios 
@app.route("/classify_user", methods=["POST"])
def classify_user():
    print("--- [DEBUG] Ruta /classify_user (GÉNERO/PERFIL) alcanzada. ---")
    
    try:
        data = request.get_json(force=True) 
        
        if not data or 'user_id' not in data:
             return jsonify({"error": "Falta user_id en el cuerpo de la solicitud."}), 400

        user_id = str(data.get("user_id"))
        user_data = ratings.get(user_id, {}) 
        num_ratings = len(user_data)
        
        # 1. LÓGICA DE CLASIFICACIÓN POR GÉNERO (Perfil de Gustos)
        
        if num_ratings >= 3:
            # Usar las preferencias de género que se calculan en /rate_song
            preferencias = user_preferences.get(user_id, {})
            
            if preferencias:
                # El primer elemento del diccionario ordenado es el favorito
                genero_favorito = list(preferencias.keys())[0]
                # Se envía el perfil de gustos
                classification_message = f"Fan de {genero_favorito}"
            else:
                classification_message = "Usuario con múltiples gustos"
                
        # 2. LÓGICA DE FALLBACK (Para nuevos o con pocas calificaciones)
        else:
            if num_ratings == 0:
                classification_message = "Nuevo usuario"
            else:
                # Si tiene 1 o 2 ratings
                classification_message = "Explorador musical"

        # Devolvemos la clasificación del perfil.
        return jsonify({
            "user_type": classification_message 
        })
        
    except Exception as e:
        print("-" * 50)
        print(f"ERROR: Falla al procesar /classify_user. Detalles: {e}")
        print("-" * 50)
        return jsonify({"error": f"Error interno del servidor. Verifique logs."}), 500
    
# Métricas del sistema 
@app.route("/metricas", methods=["GET"])
def metricas():
    if calificaciones_df is None:
        return jsonify({
            "sistema": "Modo simulación",
            "canciones_disponibles": len(songs),
            "usuarios_activos": len(ratings)
        })
    
    columnas_canciones = calificaciones_df.columns[6:]
    total_ratings = 0
    ratings_por_cancion = []
    
    for cancion_id in columnas_canciones:
        ratings_col = calificaciones_df[cancion_id]
        ratings_validas = ratings_col[ratings_col.between(1, 5)]
        if len(ratings_validas) > 0:
            avg = ratings_validas.mean()
            count = len(ratings_validas)
            ratings_por_cancion.append({
                'cancion_id': cancion_id,
                'avg_rating': round(avg, 2),
                'num_ratings': count
            })
            total_ratings += count
    
    top_canciones = sorted(ratings_por_cancion, key=lambda x: x['num_ratings'], reverse=True)[:5]
    
    return jsonify({
        "total_usuarios": len(calificaciones_df),
        "total_canciones": len(columnas_canciones),
        "total_calificaciones": total_ratings,
        "top_canciones": top_canciones
    })

# Endpoint para debug 
@app.route("/debug_user/<user_id>", methods=["GET"])
def debug_user(user_id):
    """Endpoint para ver el estado interno del usuario"""
    user_ratings = ratings.get(user_id, {})
    user_pref = user_preferences.get(user_id, "No definido")
    
    return jsonify({
        "user_id": user_id,
        "ratings": user_ratings,
        "preferencias_actuales": user_pref,
        "canciones_calificadas": [
            {
                "song_id": sid,
                "rating": r,
                "song_info": next((s for s in songs if s["id"] == sid), None)
            }
            for sid, r in user_ratings.items()
        ]
    })

# Iniciar servidor 
if __name__ == "__main__":
    print("Sistema de Recomendación de Música Iniciado")
    print("Datasets cargados:")
    print(f" - {len(songs)} canciones disponibles")
    print(f" - {len(calificaciones_df) if calificaciones_df is not None else 0} usuarios en dataset")
    print(f" - {len(popularidad_real)} canciones con datos de popularidad")
    app.run(host="0.0.0.0", port=5000, debug=True)