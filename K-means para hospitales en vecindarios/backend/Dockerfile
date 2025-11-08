FROM python:3.11-slim
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
EXPOSE 5000
# Correr la aplicación Flask de forma directa
ENV FLASK_APP=app/api.py
CMD ["python3", "-m", "app.api"]