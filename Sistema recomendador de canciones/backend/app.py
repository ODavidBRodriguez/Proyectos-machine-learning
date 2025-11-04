from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import random

app = Flask(__name__)
CORS(app)

# 🎵 Dataset simulado de canciones
songs = [
    {"id": 1, "title": "Shape of You", "artist": "Ed Sheeran", "genre": "Pop"},
    {"id": 2, "title": "Blinding Lights", "artist": "The Weeknd", "genre": "Pop"},
    {"id": 3, "title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock"},
    {"id": 4, "title": "Lose Yourself", "artist": "Eminem", "genre": "Rap"},
    {"id": 5, "title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Rock"},
    {"id": 6, "title": "Rolling in the Deep", "artist": "Adele", "genre": "Pop"},
    {"id": 7, "title": "Numb", "artist": "Linkin Park", "genre": "Rock"},
    {"id": 8, "title": "Uptown Funk", "artist": "Bruno Mars", "genre": "Funk"},
    {"id": 9, "title": "Humble", "artist": "Kendrick Lamar", "genre": "Rap"},
    {"id": 10, "title": "Billie Jean", "artist": "Michael Jackson", "genre": "Pop"},
]

# 🎧 Estructuras en memoria
ratings = {}             # user_id -> { song_id: rating }
user_preferences = {}    # user_id -> género favorito

# 🟦 Obtener canciones no calificadas
@app.route("/songs_not_rated/<user_id>", methods=["GET"])
def songs_not_rated(user_id):
    user_rated = ratings.get(user_id, {})
    not_rated = [s for s in songs if s["id"] not in user_rated]
    return jsonify(not_rated)

# 🟨 Recomendaciones personalizadas o aleatorias
@app.route("/recommendations/<user_id>", methods=["GET"])
def recommendations(user_id):
    user_ratings = ratings.get(user_id, {})
    if not user_ratings:
        recs = random.sample(songs, min(5, len(songs)))
        return jsonify(recs)

    fav_genre = user_preferences.get(user_id, None)
    if not fav_genre:
        fav_genre = "Pop"

    recs = [s for s in songs if s["genre"] == fav_genre]
    if not recs:
        recs = random.sample(songs, min(5, len(songs)))
    return jsonify(recs[:5])

# 🟩 Calificar una canción
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

    ratings[user_id][song_id] = rating

    # 🔍 Actualizar género favorito (simple promedio)
    df = pd.DataFrame([
        {"song_id": sid, "rating": r, "genre": next(s["genre"] for s in songs if s["id"] == sid)}
        for sid, r in ratings[user_id].items()
    ])
    genre_avg = df.groupby("genre")["rating"].mean().sort_values(ascending=False)
    user_preferences[user_id] = genre_avg.index[0]

    return jsonify({"message": "✅ Calificación guardada correctamente."})

# 🧩 Clasificación de usuarios (simple ejemplo)
@app.route("/classify_user", methods=["POST"])
def classify_user():
    data = request.json
    user_id = str(data.get("user_id"))
    user_data = ratings.get(user_id, {})

    if not user_data:
        return jsonify({"user_type": "Nuevo usuario"})

    high_ratings = [r for r in user_data.values() if r >= 4]
    low_ratings = [r for r in user_data.values() if r <= 2]

    if len(high_ratings) > len(low_ratings):
        return jsonify({"user_type": "Amante de la música"})
    else:
        return jsonify({"user_type": "Crítico exigente"})

# 🧮 Ejemplo de métrica de precisión (solo demostración)
@app.route("/metricas", methods=["GET"])
def metricas():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    precision = precision_score(y_true, y_pred)
    return jsonify({"precision": round(precision, 2)})

# 🚀 Iniciar servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

