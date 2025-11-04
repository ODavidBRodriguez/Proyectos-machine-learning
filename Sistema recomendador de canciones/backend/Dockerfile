FROM python:3.10-slim

WORKDIR /app

# Copiar todo el contenido del backend al contenedor
COPY . .

# ✅ Copiamos también explícitamente la carpeta modelo
COPY ./modelo /app/modelo

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
