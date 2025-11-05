from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import random
from sklearn.metrics import precision_score, f1_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Cargar datasets reales
def load_datasets():
    """Carga los datasets de canciones y calificaciones"""
    try:
        # Intentar diferentes rutas posibles
        rutas_posibles = [
            'modelo/metadata_canciones.csv',      # Ruta dentro del contenedor
            'backend/modelo/metadata_canciones.csv',  # Ruta en desarrollo
            '/app/modelo/metadata_canciones.csv',     # Ruta absoluta en contenedor
            './modelo/metadata_canciones.csv'         # Ruta relativa
        ]
        
        canciones_df = None
        calificaciones_df = None
        
        for ruta_meta in rutas_posibles:
            try:
                canciones_df = pd.read_csv(ruta_meta)
                print(f"✅ Metadata cargada desde: {ruta_meta}")
                break
            except:
                continue
        
        if canciones_df is None:
            print("❌ No se pudo cargar canciones_metadata.csv desde ninguna ruta")
            return None, None
        
        # Buscar dataset de calificaciones
        rutas_calificaciones = [
            'modelo/dataset_canciones.csv',
            'backend/modelo/dataset_canciones.csv', 
            '/app/modelo/dataset_canciones.csv',
            './modelo/dataset_canciones.csv'
        ]
        
        for ruta_cal in rutas_calificaciones:
            try:
                calificaciones_df = pd.read_csv(ruta_cal)
                print(f"✅ Calificaciones cargadas desde: {ruta_cal}")
                break
            except:
                continue
        
        if calificaciones_df is None:
            print("❌ No se pudo cargar dataset_canciones.csv desde ninguna ruta")
            return None, None
        
        print(f"📊 Datasets cargados:")
        print(f"   - Canciones: {len(canciones_df)}")
        print(f"   - Usuarios: {len(calificaciones_df)}")
        print(f"   - Calificaciones: {len(calificaciones_df.columns[6:])} canciones")
        
        return canciones_df, calificaciones_df
    
    except Exception as e:
        print(f"❌ Error cargando datasets: {e}")
        return None, None

# Cargar datos al iniciar
canciones_df, calificaciones_df = load_datasets()

# Crear lista de songs para la API
songs = []
if canciones_df is not None:
    for _, row in canciones_df.iterrows():
        songs.append({
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
    print("⚠️  Usando datos de canciones simulados")

# 🎧 Estructuras en memoria para sesiones activas
ratings = {}             # user_id -> { song_id: rating }
user_preferences = {}    # user_id -> { género: peso }

# Calcular popularidad basada en el dataset real
def calcular_popularidad_real():
    """Calcula la popularidad de cada canción basada en las calificaciones del dataset"""
    if calificaciones_df is None:
        return {}
    
    popularidad = {}
    columnas_canciones = calificaciones_df.columns[6:]  # Columnas de canciones (IDs: '1', '2', ...)
    
    for cancion_id in columnas_canciones:
        ratings_col = calificaciones_df[cancion_id]
        ratings_validas = ratings_col[ratings_col.between(1, 5)]
        
        if len(ratings_validas) > 0:
            avg_rating = ratings_validas.mean()
            num_ratings = len(ratings_validas)
            
            # Encontrar popularidad base del metadata
            cancion_meta = canciones_df[canciones_df['id'] == int(cancion_id)]
            pop_base = cancion_meta['popularidad_base'].iloc[0] if not cancion_meta.empty else 50
            
            # Fórmula de popularidad ponderada
            popularidad[cancion_id] = {
                'popularidad': (avg_rating * 15) + (num_ratings / 20) + (pop_base * 0.2),
                'avg_rating': avg_rating,
                'num_ratings': num_ratings
            }
        else:
            # Si no hay calificaciones, usar popularidad base
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
        return {"Pop": 1.0}  # Género por defecto
    
    # Agrupar calificaciones por género
    genero_ratings = {}
    for song_id, rating in user_ratings.items():
        song = next((s for s in songs if s["id"] == song_id), None)
        if song:
            genero = song["genre"]
            if genero not in genero_ratings:
                genero_ratings[genero] = []
            genero_ratings[genero].append(rating)
    
    # Calcular peso para cada género
    genero_pesos = {}
    total_peso = 0
    
    for genero, ratings_list in genero_ratings.items():
        # Promedio de calificaciones para este género
        avg_rating = sum(ratings_list) / len(ratings_list)
        
        # Cantidad de calificaciones para este género
        num_ratings = len(ratings_list)
        
        # Fórmula: (promedio * cantidad) ^ 0.7 para balancear
        peso = (avg_rating * num_ratings) ** 0.7
        genero_pesos[genero] = peso
        total_peso += peso
    
    # Normalizar pesos (que sumen 1.0)
    if total_peso > 0:
        for genero in genero_pesos:
            genero_pesos[genero] = genero_pesos[genero] / total_peso
    
    # Ordenar por peso descendente
    generos_ordenados = dict(sorted(genero_pesos.items(), key=lambda x: x[1], reverse=True))
    
    print(f"🎯 Preferencias de género para {user_id}: {generos_ordenados}")
    return generos_ordenados

@app.route("/recommendations/<user_id>", methods=["GET"])
def recommendations(user_id):
    user_ratings = ratings.get(user_id, {})
    
    # Si es usuario nuevo, recomendar por popularidad global
    if not user_ratings:
        canciones_ordenadas = sorted(
            songs, 
            key=lambda x: popularidad_real.get(str(x["id"]), {}).get('popularidad', x["popularity_base"]), 
            reverse=True
        )
        return jsonify(canciones_ordenadas[:8])  # ↑ De 5 a 8 canciones
    
    # Calcular preferencias de género dinámicamente
    generos_usuario = calcular_preferencias_genero(user_id)
    
    # Obtener recomendaciones basadas en múltiples géneros
    recs = obtener_recomendaciones_por_generos(user_id, generos_usuario)
    
    return jsonify(recs[:8])  # ↑ De 5 a 8 canciones

# También actualiza la función de recomendaciones para obtener más candidatos
def obtener_recomendaciones_por_generos(user_id, generos_usuario):
    """Obtiene recomendaciones considerando múltiples géneros"""
    user_rated = set(ratings.get(user_id, {}).keys())
    
    # Limitar a los 3 géneros principales
    generos_principales = list(generos_usuario.keys())[:3]
    
    candidatos = []
    
    for genero in generos_principales:
        peso = generos_usuario[genero]
        
        # Obtener canciones de este género que el usuario no haya calificado
        canciones_genero = [
            s for s in songs 
            if s["genre"] == genero and s["id"] not in user_rated
        ]
        
        # ↑ Aumentar de 10 a 15 canciones por género
        for cancion in canciones_genero[:15]:
            pop_score = popularidad_real.get(str(cancion["id"]), {}).get('popularidad', cancion["popularity_base"])
            score_final = pop_score * peso
            candidatos.append({
                "cancion": cancion,
                "score": score_final,
                "genero": genero,
                "peso_genero": peso
            })
    
    # Si no hay suficientes candidatos, agregar de otros géneros populares
    if len(candidatos) < 8:  # ↑ Aumentar el mínimo
        generos_restantes = [g for g in set(s["genre"] for s in songs) if g not in generos_principales]
        
        for genero in generos_restantes:
            canciones_genero = [
                s for s in songs 
                if s["genre"] == genero and s["id"] not in user_rated
            ]
            
            # ↑ Aumentar de 3 a 5 por género adicional
            for cancion in canciones_genero[:5]:
                pop_score = popularidad_real.get(str(cancion["id"]), {}).get('popularidad', cancion["popularity_base"])
                score_final = pop_score * 0.1
                candidatos.append({
                    "cancion": cancion,
                    "score": score_final,
                    "genero": genero,
                    "peso_genero": 0.1
                })
    
    # Ordenar por score y eliminar duplicados
    candidatos_ordenados = sorted(candidatos, key=lambda x: x["score"], reverse=True)
    
    # Eliminar duplicados por canción
    canciones_vistas = set()
    recomendaciones_finales = []
    
    for cand in candidatos_ordenados:
        if cand["cancion"]["id"] not in canciones_vistas:
            canciones_vistas.add(cand["cancion"]["id"])
            recomendaciones_finales.append(cand["cancion"])
        
        # ↑ Aumentar el límite para tener más opciones
        if len(recomendaciones_finales) >= 15:
            break
    
    print(f"🎵 Recomendaciones para {user_id}: {[r['title'] for r in recomendaciones_finales[:8]]}")
    return recomendaciones_finales

# 🟦 Obtener canciones no calificadas
@app.route("/songs_not_rated/<user_id>", methods=["GET"])
def songs_not_rated(user_id):
    user_rated = ratings.get(user_id, {})
    not_rated = [s for s in songs if s["id"] not in user_rated]
    
    # Mezclar todas las canciones para aleatoriedad
    canciones_mezcladas = not_rated.copy()
    random.shuffle(canciones_mezcladas)
    
    # Tomar las primeras 15 canciones mezcladas
    seleccion_aleatoria = canciones_mezcladas[:15]
    
    # Ordenar la selección por popularidad para mejor experiencia
    seleccion_ordenada = sorted(
        seleccion_aleatoria,
        key=lambda x: popularidad_real.get(str(x["id"]), {}).get('popularidad', x["popularity_base"]),
        reverse=True
    )
    
    print(f"🎵 Carrusel para {user_id}: {len(seleccion_ordenada)} canciones")
    print(f"🎭 Géneros en carrusel: {list(set(c['genre'] for c in seleccion_ordenada))}")
    
    return jsonify(seleccion_ordenada[:12])


# 🟩 Calificar una canción (ACTUALIZADA para el nuevo sistema)
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

    # Guardar la calificación
    ratings[user_id][song_id] = rating

    # Recalcular preferencias de género
    generos_usuario = calcular_preferencias_genero(user_id)
    user_preferences[user_id] = generos_usuario

    # Información de debug
    song = next((s for s in songs if s["id"] == song_id), None)
    genero_cancion = song["genre"] if song else "Desconocido"
    
    print(f"🎯 Usuario {user_id} calificó '{song['title'] if song else '?'}' ({genero_cancion}) con {rating} estrellas")
    print(f"📊 Preferencias actualizadas: {generos_usuario}")

    return jsonify({
        "message": "✅ Calificación guardada correctamente.",
        "genero_cancion": genero_cancion,
        "preferencias_actuales": generos_usuario
    })

# 🧩 Clasificación de usuarios
@app.route("/classify_user", methods=["POST"])
def classify_user():
    data = request.json
    user_id = str(data.get("user_id"))
    user_data = ratings.get(user_id, {})

    if not user_data:
        return jsonify({"user_type": "Nuevo usuario"})

    ratings_list = list(user_data.values())
    avg_rating = sum(ratings_list) / len(ratings_list) if ratings_list else 0
    num_ratings = len(user_data)
    
    if num_ratings < 3:
        return jsonify({"user_type": "Usuario en exploración"})
    elif avg_rating >= 4.0:
        return jsonify({"user_type": "Amante de la música"})
    elif avg_rating <= 2.5:
        return jsonify({"user_type": "Crítico exigente"})
    else:
        return jsonify({"user_type": "Usuario equilibrado"})

# 📊 Métricas del sistema
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

# 🐛 Endpoint para debug
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

# 🚀 Iniciar servidor
if __name__ == "__main__":
    print("🎵 Sistema de Recomendación de Música Iniciado")
    print("📊 Datasets cargados:")
    print(f"   - {len(songs)} canciones disponibles")
    print(f"   - {len(calificaciones_df) if calificaciones_df is not None else 0} usuarios en dataset")
    print(f"   - {len(popularidad_real)} canciones con datos de popularidad")
    app.run(host="0.0.0.0", port=5000, debug=True)