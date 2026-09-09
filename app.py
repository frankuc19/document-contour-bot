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

    Ambos devuelven la imagen recortada como binario (image/jpeg).
"""

from __future__ import annotations

import io
import os

import cv2
from flask import Flask, jsonify, request, send_file

from extract_document import download_image, extract_document

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok")


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
