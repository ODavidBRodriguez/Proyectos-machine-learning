const BACKEND_URL = 'https://ubicador-hospitales.onrender.com'; 

        // Paleta de colores para los clusters
        const COLORS = [
            '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A1FF33', 
            '#33A1FF', '#FF8333', '#83FF33', '#3383FF', '#FF33D6',
            '#D6FF33', '#33D6FF', '#FFC300', '#C3FF00', '#00C3FF',
            '#FF6D33', '#6DFF33', '#336DFF'
        ];

        function toggleInputs(id, isChecked) {
            // Muestra u oculta los campos de entrada basados en el checkbox de aleatoriedad
            document.getElementById(id).style.display = isChecked ? 'none' : 'block';
        }

        function ejecutarKMeans() {
            const K = parseInt(document.getElementById('num_hospitals').value);
            const resumenDiv = document.getElementById('resumen');
            resumenDiv.innerHTML = '<p class="text-indigo-600 font-medium">Calculando, por favor espere...</p>';

            const randomMatrixSize = document.getElementById('random_matrix_size').checked;
            const randomCasas = document.getElementById('random_casas').checked;
            const fijarSemilla = document.getElementById('fijar_semilla').checked;

            // Determinar el valor de la semilla
            let semilla = null;
            if (fijarSemilla) {
                // Semilla fija (42) si el usuario lo pide para reproducibilidad
                semilla = 42; 
            }

            const payload = {
                K: K,
                random_matrix_size: randomMatrixSize,
                random_casas: randomCasas,
                // 🚨 CORRECCIÓN CLAVE: INCLUIR LA SEMILLA EN EL PAYLOAD
                semilla_aleatoria: semilla 
            };

            if (!randomMatrixSize) {
                payload.M = parseInt(document.getElementById('map_size_m').value);
                payload.N = parseInt(document.getElementById('map_size_n').value);
            }

            if (!randomCasas) {
                payload.num_casas = parseInt(document.getElementById('num_casas_input').value);
            }
            
            // Validación de K
            if (K <= 0) {
                 resumenDiv.innerHTML = '<p style="color: red;">Error: K (hospitales) debe ser un número positivo.</p>';
                 return;
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
                resumenDiv.innerHTML = `<p style="color: red;">Error: ${error.message}. Asegúrese que el Backend está corriendo en <code>${BACKEND_URL}</code>.</p>`;
            });
        }

        function mostrarResultados(data) {
            const resumenDiv = document.getElementById('resumen');
            const mapaCanvas = document.getElementById('mapaCanvas');
            const clusterMapaCanvas = document.getElementById('clusterMapaCanvas'); 
            const ctxMapa = mapaCanvas.getContext('2d');
            const ctxClusterMapa = clusterMapaCanvas.getContext('2d'); 

            // Limpiar ambos canvas
            ctxMapa.clearRect(0, 0, mapaCanvas.width, mapaCanvas.height);
            ctxClusterMapa.clearRect(0, 0, clusterMapaCanvas.width, clusterMapaCanvas.height); 

            const M = data.M;
            const N = data.N;
            
            let h_list = data.hospitales.map(h => `(${h.x}, ${h.y})`).join('; ');

            if (!data.visualizacion_posible) {
                mapaCanvas.style.display = 'none';
                clusterMapaCanvas.style.display = 'none'; 
                resumenDiv.innerHTML = `
                    <h4 class="text-xl font-semibold text-red-600">¡Matriz demasiado grande para dibujar!</h4>
                    <p>Dimensiones: ${M} x ${N}. Casas: ${data.num_casas}. Se muestra solo el resumen de coordenadas.</p>
                    <h4 class="text-lg font-semibold mt-4">Hospitales (${data.K}):</h4>
                    <p class="break-all">${h_list}</p>`;
            } else {
                mapaCanvas.style.display = 'block';
                clusterMapaCanvas.style.display = 'block'; 

                // Ajustar el tamaño del canvas dinámicamente si M y N no son 100x100
                // Mantendremos un tamaño visual fijo de 400x400 y escalamos el contenido
                const canvasSize = 400;
                mapaCanvas.width = clusterMapaCanvas.width = canvasSize;
                mapaCanvas.height = clusterMapaCanvas.height = canvasSize;

                // Dibujar ambos mapas
                dibujarMapa(data, ctxMapa, canvasSize, canvasSize);
                dibujarClusterMapa(data, ctxClusterMapa, canvasSize, canvasSize);
                
                resumenDiv.innerHTML = `
                    <h4 class="text-xl font-semibold text-gray-800">Resultados de K-Means:</h4>
                    <p>Distribución óptima para <span class="font-bold text-indigo-600">${data.K}</span> hospitales en <span class="font-bold">${M}x${N}</span> (${data.num_casas} casas).</p>
                    <h4 class="text-lg font-semibold mt-4">Hospitales (${data.K}):</h4>
                    <p class="break-all">${h_list}</p>`;
            }
        }

        function dibujarMapa(data, ctx, width, height) {
            const scaleX = width / data.M;
            const scaleY = height / data.N;
            const size = Math.min(scaleX, scaleY);
            
            // Dibujar vecindarios (Color uniforme: Azul)
            ctx.fillStyle = 'blue'; 
            data.vecindarios.forEach(v => {
                ctx.beginPath();
                // Importante: La coordenada Y se invierte (height - y * scaleY) porque el canvas
                // tiene el origen (0,0) en la esquina superior izquierda.
                ctx.arc(v.x * scaleX, height - (v.y * scaleY), size * 0.15, 0, Math.PI * 2); 
                ctx.fill();
            });

            // Dibujar hospitales (Círculos grandes y rojos)
            ctx.fillStyle = 'red';
            data.hospitales.forEach(h => {
                ctx.beginPath();
                ctx.arc(h.x * scaleX, height - (h.y * scaleY), size * 0.5, 0, Math.PI * 2); 
                ctx.fill();
            });
        }

        function dibujarClusterMapa(data, ctx, width, height) {
            const scaleX = width / data.M;
            const scaleY = height / data.N;
            const size = Math.min(scaleX, scaleY);

            // Dibujar cada casa con el color de su cluster
            data.vecindarios.forEach(v => {
                const clusterColor = COLORS[v.cluster % COLORS.length]; // Usa módulo para ciclar colores
                ctx.fillStyle = clusterColor;
                ctx.beginPath();
                ctx.arc(v.x * scaleX, height - (v.y * scaleY), size * 0.15, 0, Math.PI * 2); 
                ctx.fill();
            });

            // Dibujar los hospitales (Color: Negro)
            ctx.fillStyle = 'black'; 
            data.hospitales.forEach(h => {
                ctx.beginPath();
                ctx.arc(h.x * scaleX, height - (h.y * scaleY), size * 0.5, 0, Math.PI * 2); 
                ctx.fill();
            });
        }