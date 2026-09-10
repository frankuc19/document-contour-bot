"""
Microservicio HTTP que envuelve extract_document.py para poder llamarlo
desde un workflow de n8n (nodo HTTP Request), ya que el Code node de n8n
no puede correr OpenCV directamente (ni en modo JavaScript ni en el modo
Python/Pyodide).

Uso local:
    python app.py
    # sirve en http://localhost:8000

Endpoints:
    GET  /health
    POST /extract   body JSON: {"url": "https://..."}
    GET  /extract?url=https://...
        Devuelve la imagen recortada como binario (image/jpeg).

    POST /badge      body JSON: {"html": "<div>...</div>"}
        Renderiza el HTML con Chromium (Playwright), sube el PNG resultante
        a Supabase Storage y devuelve {"url": "https://.../gafetes/xxx.png"}.
        Requiere las variables de entorno SUPABASE_URL, SUPABASE_KEY y
        SUPABASE_BUCKET (ver .env.example).
"""

from __future__ import annotations

import io
import os
import threading
import uuid

import cv2
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from playwright.sync_api import sync_playwright
from supabase import Client, create_client

from extract_document import download_image, extract_document

load_dotenv()

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "gafetes")

_playwright = None
_browser = None
_browser_lock = threading.Lock()

_supabase: Client | None = None


def get_browser():
    """Lanza Chromium una sola vez por proceso worker y lo reutiliza."""
    global _playwright, _browser
    with _browser_lock:
        if _browser is None:
            _playwright = sync_playwright().start()
            _browser = _playwright.chromium.launch(args=["--no-sandbox"])
    return _browser


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "Faltan las variables de entorno SUPABASE_URL / SUPABASE_KEY"
            )
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/badge")
def badge():
    data = request.get_json(silent=True) or {}
    html = data.get("html")
    if not html:
        return jsonify(error="Falta 'html'"), 400

    # Envuelve el HTML recibido en un contenedor que se ajusta a su
    # contenido (shrink-wrap), para que el screenshot recorte justo al
    # tamaño del gafete y no a todo el viewport.
    wrapped_html = (
        "<html><head><style>"
        "html,body{margin:0;padding:0;background:transparent;}"
        "</style></head><body>"
        '<div id="__badge_root" style="display:inline-block;">'
        f"{html}</div></body></html>"
    )

    try:
        browser = get_browser()
        page = browser.new_page(viewport={"width": 1200, "height": 1200})
        try:
            page.set_content(wrapped_html, wait_until="networkidle")
            png_bytes = page.locator("#__badge_root").screenshot(
                omit_background=True
            )
        finally:
            page.close()
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"No se pudo renderizar el HTML: {exc}"), 500

    filename = f"{uuid.uuid4()}.png"
    try:
        supabase = get_supabase()
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            filename, png_bytes, {"content-type": "image/png"}
        )
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(
            filename
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"No se pudo subir la imagen a Supabase: {exc}"), 500

    return jsonify(url=public_url)


@app.route("/extract", methods=["GET", "POST"])
def extract():
    url = request.args.get("url")
    if not url and request.is_json:
        url = (request.get_json(silent=True) or {}).get("url")

    if not url:
        return jsonify(error="Falta el parámetro 'url'"), 400

    try:
        image = download_image(url)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"No se pudo descargar la imagen: {exc}"), 400

    try:
        result = extract_document(image)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"No se pudo procesar la imagen: {exc}"), 500

    ok, buffer = cv2.imencode(".jpg", result)
    if not ok:
        return jsonify(error="No se pudo codificar la imagen resultante"), 500

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="foto.jpg",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
