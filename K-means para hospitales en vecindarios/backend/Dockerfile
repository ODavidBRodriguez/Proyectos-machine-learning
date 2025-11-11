FROM python:3.11-slim
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
EXPOSE 8000
# Correr la aplicación Flask de forma directa
CMD ["gunicorn", "app.api:app", "--bind", "0.0.0.0:8000"]