# Usamos una imagen ligera de Nginx para servir archivos estáticos
FROM nginx:alpine


# Copiamos nuestros archivos estáticos al directorio de contenido web de Nginx
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY script.js /usr/share/nginx/html/ 


EXPOSE 80