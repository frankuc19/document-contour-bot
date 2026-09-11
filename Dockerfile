# Imagen oficial de Playwright: ya trae Chromium + todas las librerías de
# sistema que necesita preinstaladas. El builder nativo de Render (sin
# Docker) no da acceso root, así que `playwright install --with-deps` falla
# ahí (necesita sudo/apt-get). Con Docker, el build corre como root y no hay
# ese problema.
#
# La versión de la imagen debe coincidir EXACTO con la versión del paquete
# "playwright" fijada en requirements.txt (Playwright valida que la versión
# del cliente Python coincida con el binario del navegador instalado).
FROM mcr.microsoft.com/playwright/python:v1.62.0-noble

# Aseguramos root explícitamente para el build y para lanzar Chromium sin
# problemas de sandboxing (app.py ya pasa --no-sandbox al lanzar el browser).
USER root

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render inyecta el puerto real en $PORT; el 8000 es solo el default para
# correr el contenedor localmente.
ENV PORT=8000
EXPOSE 8000

# --timeout 90: margen extra sobre el default de gunicorn (30s). La
# búsqueda de rostro en varios ángulos (extract_document.py) ya se
# optimizó para ser rápida, pero en el CPU compartido de Render un caso
# lento no debería morir con 500 por el timeout por defecto.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8000} --timeout 90"]
