import pandas as pd
import numpy as np

# --- 1. CONSTANTES DE MAPEO ---

# Nombres de columna originales (corruptos) a nombres estandarizados y limpios
MAPPING_NOMBRES_COLUMNA = {
    # 🛑 Cambia estos patrones para que coincidan EXACTAMENTE con lo que ves después de la carga 🛑
    'Ã¯Â»Â¿UserID': 'UserID_Limpio',
    'GÃ\x83Â\x83Ã\x82Â©nero': 'Genero',          # Patrón corregido (vs. GÃƒÂƒÃ‚Â©nero)
    'RegiÃ\x83Â\x83Ã\x82Â³n': 'Region',          # Patrón corregido (vs. RegiÃƒÂƒÃ‚Â³n)
    'GÃ\x83Â\x83Ã\x82Â©neroFav': 'GeneroFavorito', # Patrón corregido
    'ClaseUsuario': 'ClaseUsuario',
}

# EN limpieza_datos_fuente.py, dentro de la constante MAPEO_VALORES_PERFIL

MAPEO_VALORES_PERFIL = {
    # --- REGION ---
    # 🛑 CORRECCIÓN: Usamos el patrón exacto en minúsculas 🛑
    'bogotãââ¡': 'Bogota', 
    'pacãââ­fico': 'Pacifico',

    # --- CLASE DE USUARIO ---
    # 🛑 CORRECCIÓN: Usamos el patrón exacto en minúsculas 🛑
    'salsãââ³mano': 'Salsamano', 
    
    # --- OTROS GÉNEROS / PERFILES ---
    'folclor': 'Cumbia', 
    'folclorico': 'Cumbia',
    'urbano': 'Reggaeton', 
    'folclorista': 'Amante del Vallenato',
    'urbano': 'Reggaetonero', 
}

# --- 2. FUNCIÓN DE LIMPIEZA PRINCIPAL ---

def limpiar_dataset_calificaciones(archivo_entrada, archivo_salida):
    try:
        df = pd.read_csv(archivo_entrada, encoding='latin-1')
        print(df.columns.tolist())
        print(f"✅ Archivo '{archivo_entrada}' cargado.")

        print("\n--- Primeros 20 Registros del DataFrame Cargado (para verificar codificación) ---")
        print(df.head(20).to_string()) # Usamos .to_string() para que se imprima completo en la terminal
        print("--------------------------------------------------------------------------------\n")
    except Exception as e:
        print(f"❌ ERROR durante la carga: {e}"); return

    # --- A. LIMPIEZA DE NOMBRES DE COLUMNA ---
    df.rename(columns=MAPPING_NOMBRES_COLUMNA, inplace=True)
    print("✅ Nombres de columna estandarizados y corregidos.")

    # Corregir el BOM de las IDs de Canción (Columnas 1, 2, 3...)
    columnas_a_renombrar = {}
    for col in df.columns:
        if isinstance(col, str) and col.startswith('ï»¿') and col[1:].isdigit():
            columnas_a_renombrar[col] = col[1:] # Ej: 'ï»¿1' a '1'

    if columnas_a_renombrar:
        df.rename(columns=columnas_a_renombrar, inplace=True)
        print("✅ BOM de ID de Canción en las columnas corregido.")


    # --- B. LIMPIEZA DE VALORES DE PERFIL (CONSISTENCIA) ---

    def mapear_valor_celda(valor, columna):
        """Aplica mapeo y limpieza a cada celda de perfil."""
        if isinstance(valor, str):
            valor_stripped = valor.strip()
            valor_lower = valor_stripped.lower()
            
            # Aplicar mapeo de neutralización/corrección
            if valor_lower in MAPEO_VALORES_PERFIL:
                # 🛑 CLAVE 1: Retorna el valor LIMPIO del diccionario. 
                # Esto corrige el valor y el acento a la vez.
                return MAPEO_VALORES_PERFIL[valor_lower]
            
            # Si no hay mapeo, devuelve el valor original con espacios eliminados.
            # Si la celda es 'Caribe' (limpia), devolverá 'Caribe'.
            # Si la celda es 'BogotÃÂÃÂ¡' y no se mapeó, la devuelve corrupta.
            return valor_stripped 
    
    # Aplicar la limpieza a las columnas de perfil de interés (usando los nuevos nombres)
    columnas_perfil_a_limpiar = ['Region', 'GeneroFavorito', 'ClaseUsuario']
    
    for col in columnas_perfil_a_limpiar:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: mapear_valor_celda(x, col))
            
    print("✅ Valores de perfil inconsistentes (Folclor, Urbano) mapeados.")

    # --- C. LIMPIEZA DE TIPOS DE DATOS Y GUARDADO ---

    # Identificar la posición de inicio de las calificaciones
    indice_inicio_calificaciones = df.columns.get_loc('ClaseUsuario') + 1
    columnas_calificaciones = df.columns[indice_inicio_calificaciones:]
    
    # Forzar las calificaciones a tipo numérico y rellenar NaN con 0.0
    df[columnas_calificaciones] = df[columnas_calificaciones].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0.0)
    
    print("✅ Calificaciones forzadas a numérico y NaN a 0.0.")


    # Guardar el DataFrame limpio
    df.to_csv(archivo_salida, index=False, encoding='utf-8')
    print(f"\n🎉 ¡Limpieza Finalizada! Se generó el archivo limpio: '{archivo_salida}'")

# --- 3. EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    limpiar_dataset_calificaciones('backend\modelo\dataset_canciones.csv', 'dataset_canciones.csv')