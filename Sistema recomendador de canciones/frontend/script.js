
const API_URL = window.location.hostname === "localhost"
  ? "http://localhost:5000"
  : "http://music_backend:5000"

let USER_ID = localStorage.getItem('currentUserId');
if (!USER_ID) {
    // Generar un ID único simple si no existe
    USER_ID = 'user_' + Date.now();
    localStorage.setItem('currentUserId', USER_ID);
    console.log("Nuevo usuario generado:", USER_ID); 
}

// Cargar y mostrar la clasificación del usuario
async function loadUserClassification() {
    try {
        const res = await fetch(`${API_URL}/classify_user`, {
            method: "POST", 
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: USER_ID })
        });
        
        if (!res.ok) { 
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const data = await res.json();
        
        const classificationElement = document.getElementById("user-classification");
        if (classificationElement) {
            
            // Espera el campo 'user_type' de la respuesta
            const message = `Hola, ${data.user_type}!`;
            
            classificationElement.textContent = message;
        }
    } catch (err) {
        console.error("Error cargando clasificación:", err); 
    }
}


// Función para configurar los event listeners de los botones
function setupRateButtons() {
    // Event delegation para los botones de calificar
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('rate-btn')) {
            const songId = e.target.getAttribute('data-song-id');
            const songTitle = e.target.closest('.carousel-item').querySelector('h5').textContent;
            const songArtist = e.target.closest('.carousel-item').querySelector('div').textContent.split(' - ')[0];
            showRatingView(songId, songTitle, songArtist);
        }
    });
}

async function loadSongs() {
  try {
    const res = await fetch(`${API_URL}/songs_not_rated/${USER_ID}`);
    const data = await res.json();
    const carousel = document.getElementById("carousel-content");

    if (!carousel) return;

    carousel.innerHTML = "";

    data.forEach((song, index) => {
      const activeClass = index === 0 ? "active" : "";
      const div = document.createElement("div");
      div.className = `carousel-item ${activeClass}`;
      div.innerHTML = `
        <div class="text-center p-4">
          <h5 style="color: white; font-weight: bold;">${song.title}</h5>
          <div style="color: white; opacity: 0.9; margin: 10px 0;">
              ${song.artist} - ${song.genre}
          </div>
          <button class="rate-btn" onclick="goToRating(${song.id})">
              Calificar canción
          </button>
        </div>
      `;
      carousel.appendChild(div);
    });
  } catch (err) {
    console.error("Error cargando canciones:", err);
  }
}

// Redirigir a la vista de calificación
function goToRating(songId) {
  window.location.href = `calificacion.html?song_id=${songId}`;
}

// Cargar recomendaciones
async function loadRecommendations() {
  try {
    const res = await fetch(`${API_URL}/recommendations/${USER_ID}`);
    const data = await res.json();
    const container = document.getElementById("recommended");

    if (!container) return;

    container.innerHTML = "";
    data.forEach(song => {
      const div = document.createElement("div");
      div.className = "col-md-4 mb-3";
      div.innerHTML = `
        <div class="card p-3 shadow-sm h-100">
          <h6>${song.title}</h6>
          <p class="text-muted">${song.artist} - ${song.genre}</p>
        </div>
      `;
      container.appendChild(div);
    });
  } catch (err) {
    console.error("Error cargando recomendaciones:", err);
  }
}

// Enviar calificación
async function sendRating() {
  const urlParams = new URLSearchParams(window.location.search);
  const songId = urlParams.get("song_id");
  const rating = parseInt(document.getElementById("rating").value);

  if (rating < 1 || rating > 5) {
    alert("Por favor selecciona una calificación entre 1 y 5.");
    return;
  }

  try {
    const res = await fetch(`${API_URL}/rate_song`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID, song_id: songId, rating })
    });

    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }

    //  Mostrar mensaje y volver al inicio
    localStorage.setItem("alerta", "true");
    window.location.href = "index.html";
  } catch (err) {
    console.error("Error enviando calificación:", err);
  }
}

// Mostrar alerta al volver al inicio
function showAlertIfNeeded() {
  const alertBox = document.getElementById("alerta");
  if (localStorage.getItem("alerta") === "true" && alertBox) {
    alertBox.classList.remove("d-none");
    setTimeout(() => {
      alertBox.classList.add("d-none");
      localStorage.removeItem("alerta");
    }, 3000);
  }
}

// Inicialización automática
document.addEventListener("DOMContentLoaded", () => {
  if (window.location.pathname.endsWith("index.html") || window.location.pathname === "/") {
    showAlertIfNeeded();
    loadUserClassification();
    loadSongs();
    loadRecommendations();
  }
});