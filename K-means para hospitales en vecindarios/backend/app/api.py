from flask import Flask, request, jsonify
from .localizar_hospital import LocalizarHospital 
from flask_cors import CORS 

app = Flask(__name__)
# CORS habilitado para comunicación con el frontend
CORS(app) 

@app.route('/api/locate', methods=['POST'])
def locate_hospitals():
    
    try:
        data = request.get_json()
        
        K = int(data.get('K', 3))
        
        # Parámetros condicionales
        M = int(data.get('M', 100))
        N = int(data.get('N', 100))
        num_casas = int(data.get('num_casas', 500))
        
        # Flags booleanos para aleatoriedad
        random_matrix_size = bool(data.get('random_matrix_size', False))
        random_casas = bool(data.get('random_casas', False))

        semilla_casas = data.get('semilla_aleatoria')

        # Validación básica
        if K <= 0 or (M <= 0 and not random_matrix_size) or (N <= 0 and not random_matrix_size):
            return jsonify({'error': 'Todos los parámetros deben ser positivos.'}), 400

        # Se instancia el localizador con todos los parámetros y flags
        locator = LocalizarHospital(
            K=K, 
            M=M, 
            N=N, 
            num_casas=num_casas,
            random_matrix_size=random_matrix_size,
            random_casas=random_casas,
            semilla_casas=semilla_casas
        )
        
        # El método ubicar_hospitales se encarga del K-Means
        locator.ubicar_hospitales()
        
        
        results = locator.obtener_resultados()
        
        return jsonify(results), 200

    except ValueError as e:
        return jsonify({'error': f'Error de validación: {e}'}), 400
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        return jsonify({'error': 'Error interno del servidor al ejecutar K-Means.'}), 500

# Recuerda que la ejecución es por Docker CMD: python3 -m app.api
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
