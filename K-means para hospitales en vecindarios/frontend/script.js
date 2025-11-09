const BACKEND_URL = 'http://localhost:5001/api/locate'; 

// Paleta de colores atractiva para los clusters
// Aseguramos que haya suficientes colores para K clusters
const COLORS = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A1FF33', 
    '#33A1FF', '#FF8333', '#83FF33', '#3383FF', '#FF33D6',
    '#D6FF33', '#33D6FF', '#FFC300', '#C3FF00', '#00C3FF',
    '#FF6D33', '#6DFF33', '#336DFF'
];

function toggleInputs(id, isChecked) {
    document.getElementById(id).style.display = isChecked ? 'none' : 'block';
}

function ejecutarKMeans() {
    const K = parseInt(document.getElementById('num_hospitals').value);
    const resumenDiv = document.getElementById('resumen');
    resumenDiv.innerHTML = 'Calculando...';

    const randomMatrixSize = document.getElementById('random_matrix_size').checked;
    const randomCasas = document.getElementById('random_casas').checked;

    const payload = {
        K: K,
        random_matrix_size: randomMatrixSize,
        random_casas: randomCasas
    };

    if (!randomMatrixSize) {
        payload.M = document.getElementById('map_size_m').value;
        payload.N = document.getElementById('map_size_n').value;
    }

    if (!randomCasas) {
        payload.num_casas = document.getElementById('num_casas_input').value;
    }

    fetch(BACKEND_URL, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error || `Error HTTP: ${response.status}`); });
        }
        return response.json();
    })
    .then(data => {
        mostrarResultados(data);
    })
    .catch(error => {
        console.error('Error al conectar/procesar:', error);
        resumenDiv.innerHTML = `<p style="color: red;">Error: ${error.message}. Asegúrese que el Backend está en ${BACKEND_URL}.</p>`;
    });
}

function mostrarResultados(data) {
    const resumenDiv = document.getElementById('resumen');
    const mapaCanvas = document.getElementById('mapaCanvas');
    const clusterMapaCanvas = document.getElementById('clusterMapaCanvas'); // Nuevo canvas
    const ctxMapa = mapaCanvas.getContext('2d');
    const ctxClusterMapa = clusterMapaCanvas.getContext('2d'); // Contexto del nuevo canvas

    ctxMapa.clearRect(0, 0, mapaCanvas.width, mapaCanvas.height);
    ctxClusterMapa.clearRect(0, 0, clusterMapaCanvas.width, clusterMapaCanvas.height); // Limpiar también el nuevo

    const M = data.M;
    const N = data.N;
    
    let h_list = data.hospitales.map(h => `(${h.x}, ${h.y})`).join('; ');

    if (!data.visualizacion_posible) {
        mapaCanvas.style.display = 'none';
        clusterMapaCanvas.style.display = 'none'; // Ocultar también el nuevo
        resumenDiv.innerHTML = `
            <h4>¡Matriz demasiado grande para dibujar!</h4>
            <p>Dimensiones: ${M} x ${N}. Casas: ${data.num_casas}. Se muestra solo el resumen de coordenadas.</p>
            <h4>Hospitales (${data.K}):</h4>
            <p>${h_list}</p>`;
    } else {
        mapaCanvas.style.display = 'block';
        clusterMapaCanvas.style.display = 'block'; // Mostrar también el nuevo

        // Dibujar ambos mapas
        dibujarMapa(data, ctxMapa, mapaCanvas.width, mapaCanvas.height);
        dibujarClusterMapa(data, ctxClusterMapa, clusterMapaCanvas.width, clusterMapaCanvas.height);
        
        resumenDiv.innerHTML = `
            <h4>Resultados de K-Means:</h4>
            <p>Distribución óptima para ${data.K} hospitales en ${M}x${N} (${data.num_casas} casas).</p>
            <h4>Hospitales (${data.K}):</h4>
            <p>Coordenadas: ${h_list}</p>`;
    }
}

function dibujarMapa(data, ctx, width, height) {
    const scaleX = width / data.M;
    const scaleY = height / data.N;

    // Dibujar vecindarios (ahora los cargamos del nuevo formato)
    ctx.fillStyle = 'blue'; // Color uniforme para este mapa
    data.vecindarios.forEach(v => {
        ctx.beginPath();
        ctx.arc(v.x * scaleX, height - (v.y * scaleY), 1.5, 0, Math.PI * 2); 
        ctx.fill();
    });

    // Dibujar hospitales (círculos grandes y rojos)
    ctx.fillStyle = 'red';
    data.hospitales.forEach(h => {
        ctx.beginPath();
        ctx.arc(h.x * scaleX, height - (h.y * scaleY), 5, 0, Math.PI * 2); 
        ctx.fill();
    });
}

// 🚨 Nueva función para dibujar los clusters 🚨
function dibujarClusterMapa(data, ctx, width, height) {
    const scaleX = width / data.M;
    const scaleY = height / data.N;

    // Dibujar cada casa con el color de su cluster
    data.vecindarios.forEach(v => {
        const clusterColor = COLORS[v.cluster % COLORS.length]; // Usa módulo para ciclar colores
        ctx.fillStyle = clusterColor;
        ctx.beginPath();
        ctx.arc(v.x * scaleX, height - (v.y * scaleY), 1.5, 0, Math.PI * 2); 
        ctx.fill();
    });

    // Opcional: Podrías dibujar también los hospitales en este mapa, si quieres ver su relación con los clusters.
    ctx.fillStyle = 'black'; // Un color diferente para distinguirlos
    data.hospitales.forEach(h => {
    ctx.beginPath();
    ctx.arc(h.x * scaleX, height - (h.y * scaleY), 5, 0, Math.PI * 2); 
    ctx.fill();
    });
}