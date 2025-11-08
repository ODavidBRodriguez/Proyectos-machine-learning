// Llamada directa al puerto del host (5001) para evitar problemas de proxy/CORS
const BACKEND_URL = 'http://localhost:5001/api/locate'; 

// Función para mostrar/ocultar los campos de entrada
function toggleInputs(id, isChecked) {
    document.getElementById(id).style.display = isChecked ? 'none' : 'block';
}

function ejecutarKMeans() {
    const K = parseInt(document.getElementById('num_hospitals').value);
    const resumenDiv = document.getElementById('resumen');
    resumenDiv.innerHTML = 'Calculando...';

    // 1. Obtener los flags de aleatoriedad
    const randomMatrixSize = document.getElementById('random_matrix_size').checked;
    const randomCasas = document.getElementById('random_casas').checked;

    const payload = {
        K: K,
        // Flags que le dicen al backend cómo generar los datos
        random_matrix_size: randomMatrixSize,
        random_casas: randomCasas
    };

    // 2. Añadir M, N y num_casas solo si NO son aleatorios
    if (!randomMatrixSize) {
        // Se envían como texto, pero el backend los parsea a entero
        payload.M = document.getElementById('map_size_m').value;
        payload.N = document.getElementById('map_size_n').value;
    }

    if (!randomCasas) {
        payload.num_casas = document.getElementById('num_casas_input').value;
    }

    // 3. Enviar la petición con el payload dinámico
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
    const canvas = document.getElementById('mapaCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // M y N se reciben del Backend (el valor final, incluso si fue aleatorio)
    const M = data.M;
    const N = data.N;
    
    let h_list = data.hospitales.map(h => `(${h.x}, ${h.y})`).join('; ');

    if (!data.visualizacion_posible) {
        // Caso Matriz Grande
        canvas.style.display = 'none';
        resumenDiv.innerHTML = `
            <h4>¡Matriz Demasiado Grande para Dibujar!</h4>
            <p>Dimensiones: ${M} x ${N}. Vecindarios: ${data.num_casas}. Se muestra solo el resumen de coordenadas.</p>
            <h4>Hospitales (${data.K}):</h4>
            <p>${h_list}</p>`;
    } else {
        // Caso Matriz Pequeña
        canvas.style.display = 'block';
        dibujarMapa(data, ctx, canvas.width, canvas.height);
        resumenDiv.innerHTML = `
            <h4>Resultados de K-Means:</h4>
            <p>Distribución óptima para ${data.K} hospitales en **${M}x${N}** (${data.num_casas} vecindarios).</p>
            <h4>Hospitales (${data.K}):</h4>
            <p>${h_list}</p>`;
    }
}

function dibujarMapa(data, ctx, width, height) {
    const scaleX = width / data.M;
    const scaleY = height / data.N;

    // Dibujar vecindarios (puntos azules)
    ctx.fillStyle = 'blue';
    data.vecindarios.forEach(v => {
        ctx.beginPath();
        // Nota: Se invierte Y para que 0,0 esté en la esquina inferior izquierda
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