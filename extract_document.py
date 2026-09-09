"""
Descarga una imagen desde una URL (p. ej. un link de S3 de un documento tipo
licencia/credencial) y "rescata" la foto de la persona que está dentro del
documento: busca el contorno rectangular tipo foto-carnet (tamaño intermedio,
proporción retrato) y recorta solo esa región.

Uso:
    python extract_document.py <url> [--output salida.jpg] [--debug]

    # Modo lote: un archivo de texto con una URL por línea
    python extract_document.py --url-file urls.txt --output-dir salidas/

Requisitos: ver requirements.txt (pip install -r requirements.txt)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

REQUEST_TIMEOUT = 30
# Ancho al que se reescala la imagen para detectar bordes (más rápido y
# estable); el recorte final se hace sobre la imagen original a resolución
# completa usando la razón de escala.
PROCESSING_WIDTH = 1000


def download_image(url: str) -> np.ndarray:
    """Descarga una imagen desde una URL y la decodifica con OpenCV (BGR)."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo decodificar la imagen descargada de: {url}")
    return image


def resize_for_processing(image: np.ndarray, width: int = PROCESSING_WIDTH):
    h, w = image.shape[:2]
    ratio = w / float(width) if w > width else 1.0
    if ratio == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(image, (int(w / ratio), int(h / ratio)))
    return resized, ratio


# Rango de proporción (ancho/alto) típico de una foto tipo carnet/pasaporte
# en retrato (p. ej. ~3:4 = 0.75). Se deja holgado para tolerar variaciones.
PHOTO_ASPECT_RANGE = (0.55, 1.15)
PHOTO_ASPECT_IDEAL = 0.78
# Una foto de credencial suele ocupar entre ~2% y ~45% del área total del
# documento (documentos muy recortados alrededor de la foto pueden llegar
# más alto, de ahí el margen amplio).
PHOTO_AREA_RANGE = (0.02, 0.45)
# Qué tan "rectangular" debe ser el contorno (area del contorno / area del
# bounding box). Las fotos con esquinas redondeadas siguen siendo altas.
MIN_EXTENT = 0.55
# Padding relativo agregado alrededor del recorte final.
CROP_PADDING_RATIO = 0.02


def find_photo_region(processed_image: np.ndarray):
    """
    Busca, dentro de la imagen del documento (ya reescalada), el contorno
    rectangular que más se parece a una foto tipo carnet: no es el borde
    completo del documento, sino una región interior de tamaño intermedio
    y proporción retrato. Devuelve (x, y, w, h, todos_los_contornos) o
    (None, todos_los_contornos) si no encuentra un candidato razonable.
    """
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Umbral de Canny automático basado en la mediana de intensidad.
    median = np.median(gray)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = processed_image.shape[0] * processed_image.shape[1]
    min_aspect, max_aspect = PHOTO_ASPECT_RANGE
    min_area_ratio, max_area_ratio = PHOTO_AREA_RANGE

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue

        # Canny suele trazar el marco de la foto como una línea delgada
        # (doble borde), lo que hace que contourArea() dé un área casi nula
        # aunque el contorno recorra todo el rectángulo. El área del convex
        # hull sí aproxima bien el área real encerrada por ese marco.
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        area_ratio = area / float(image_area)
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        aspect = w / float(h)
        if not (min_aspect <= aspect <= max_aspect):
            continue

        extent = area / float(w * h)
        if extent < MIN_EXTENT:
            continue

        # Prioriza contornos grandes cuya proporción se acerque a la de una
        # foto de credencial típica.
        aspect_penalty = abs(aspect - PHOTO_ASPECT_IDEAL)
        score = area * (1.0 / (1.0 + aspect_penalty))
        candidates.append((score, x, y, w, h))

    if not candidates:
        return None, contours

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, x, y, w, h = candidates[0]
    return (x, y, w, h), contours


def extract_document(image: np.ndarray, debug_path: Path | None = None) -> np.ndarray:
    """Detecta la foto de la persona dentro del documento y la recorta."""
    processed, ratio = resize_for_processing(image)
    box, all_contours = find_photo_region(processed)

    if debug_path is not None:
        debug_img = processed.copy()
        cv2.drawContours(debug_img, all_contours, -1, (0, 255, 0), 1)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_img)

    if box is None:
        print(
            "  ! No se detectó una foto de credencial dentro del documento; "
            "se devuelve la imagen completa.",
            file=sys.stderr,
        )
        return image

    # Escala el recuadro de vuelta a la resolución original y agrega un
    # pequeño padding para no cortar el borde de la foto.
    x, y, w, h = box
    x, y, w, h = x * ratio, y * ratio, w * ratio, h * ratio
    pad_x, pad_y = w * CROP_PADDING_RATIO, h * CROP_PADDING_RATIO

    img_h, img_w = image.shape[:2]
    x0 = max(0, int(x - pad_x))
    y0 = max(0, int(y - pad_y))
    x1 = min(img_w, int(x + w + pad_x))
    y1 = min(img_h, int(y + h + pad_y))

    return image[y0:y1, x0:x1]


def filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name if name else "documento.jpg"


def process_one(url: str, output_path: Path, debug: bool) -> None:
    print(f"Descargando: {url}")
    image = download_image(url)

    debug_path = output_path.with_name(output_path.stem + "_debug" + output_path.suffix) if debug else None
    result = extract_document(image, debug_path=debug_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"  -> Guardado en: {output_path}")
    if debug_path is not None:
        print(f"  -> Debug (contornos detectados): {debug_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="URL de la imagen a procesar")
    parser.add_argument(
        "--url-file",
        type=Path,
        help="Archivo de texto con una URL por línea (modo lote)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("salida.jpg"),
        help="Ruta de salida para una sola imagen (default: salida.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("salidas"),
        help="Carpeta de salida para modo lote (default: salidas/)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Guarda también una imagen con los contornos detectados dibujados",
    )
    args = parser.parse_args()

    if not args.url and not args.url_file:
        parser.error("Debes indicar una URL o --url-file")

    if args.url_file:
        urls = [
            line.strip()
            for line in args.url_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for url in urls:
            out_path = args.output_dir / filename_from_url(url)
            try:
                process_one(url, out_path, args.debug)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! Error procesando {url}: {exc}", file=sys.stderr)
    else:
        process_one(args.url, args.output, args.debug)


if __name__ == "__main__":
    main()
