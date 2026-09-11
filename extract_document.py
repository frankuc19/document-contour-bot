"""
Descarga una imagen desde una URL (p. ej. un link de S3 de un documento tipo
licencia/credencial) y "rescata" la foto de la persona que está dentro del
documento: primero busca la cara con el detector de rostros de OpenCV y
recorta un encuadre tipo retrato alrededor de ella. Si no encuentra una cara
con suficiente confianza, cae de respaldo a un heurístico de contorno
(recuadro rectangular tipo foto-carnet, descartando QR/códigos de barras).

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
# Densidad máxima de píxeles de borde dentro del recuadro (proporción de
# píxeles Canny "encendidos"). Un código QR/barras es un patrón de alta
# frecuencia con densidad muy alta (~0.5+); una foto de persona es mucho
# más lisa (~0.1-0.2). Este filtro evita confundir un QR con la foto.
MAX_EDGE_DENSITY = 0.35
# Padding relativo agregado alrededor del recorte final (método de contorno).
CROP_PADDING_RATIO = 0.02

# --- Detección de rostro (método principal) ---------------------------

# Confianza mínima (levelWeight de Haar Cascade) para aceptar una cara como
# la foto del documento y no una falsa detección (fondo, mano, textura).
# Calibrado con fotos reales: caras verdaderas de credencial suelen dar
# ~9-11, falsos positivos de fondo/mano suelen dar <3.
MIN_FACE_CONFIDENCE = 4.0
# Márgenes para expandir la caja de la cara (detectada muy ajustada, solo
# ojos-nariz-boca) a un encuadre tipo retrato de credencial (cabeza+hombros).
# Relativos al ancho/alto de la caja de la cara.
FACE_MARGIN_LEFT = 0.6
FACE_MARGIN_RIGHT = 0.5
FACE_MARGIN_TOP = 0.35
FACE_MARGIN_BOTTOM = 0.75

_face_cascade: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def find_face_box(processed_image: np.ndarray):
    """
    Busca la cara más confiable en la imagen (ya reescalada) usando Haar
    Cascade. Devuelve (x, y, w, h, confianza) de la mejor detección, o
    None si no hay ninguna.
    """
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape[:2]
    min_size = (max(20, int(w * 0.03)), max(20, int(h * 0.03)))

    cascade = _get_face_cascade()
    rects, _reject_levels, level_weights = cascade.detectMultiScale3(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=min_size,
        outputRejectLevels=True,
    )

    if len(rects) == 0:
        return None

    best_idx = int(np.argmax(level_weights))
    x, y, fw, fh = rects[best_idx]
    return int(x), int(y), int(fw), int(fh), float(level_weights[best_idx])


# Ángulos (en grados) que se prueban cuando la cara no aparece derecha: fotos
# tomadas con el documento girado (sobre una mesa, en la mano, etc.) hacen
# que Haar Cascade falle porque no tolera más de ~15-20° de inclinación.
FACE_SEARCH_ANGLES = (
    0,
    90,
    -90,
    180,
    10,
    -10,
    20,
    -20,
    30,
    -30,
    40,
    -40,
    50,
    -50,
)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rota la imagen `angle` grados alrededor de su centro sin recortarla
    (expande el lienzo para que no se pierdan las esquinas)."""
    if angle == 0:
        return image

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image, matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE
    )


# Ancho usado para el barrido rápido de ángulos (ver find_best_face). Correr
# Haar Cascade en los 14 ángulos a resolución completa (PROCESSING_WIDTH)
# era demasiado lento en el CPU compartido de Render: superaba el timeout
# de gunicorn (~30s) y el request moría con 500/502. A esta resolución
# mucho más chica, el barrido completo es barato; el ángulo ganador se
# vuelve a probar una sola vez a resolución completa para un recuadro
# preciso.
COARSE_SEARCH_WIDTH = 350


def find_best_face(processed_image: np.ndarray):
    """
    Encuentra el ángulo/recuadro de la cara con más confianza, probando
    varias rotaciones. Devuelve (angulo, x, y, w, h, ancho_rotado,
    alto_rotado) en coordenadas de `processed_image` rotada ese ángulo, o
    None si no se encontró ninguna cara confiable en ningún ángulo.
    """
    coarse_image, _ratio = resize_for_processing(
        processed_image, width=COARSE_SEARCH_WIDTH
    )

    coarse_best = None  # (weight, angle)
    for angle in FACE_SEARCH_ANGLES:
        rotated = rotate_image(coarse_image, angle)
        result = find_face_box(rotated)
        if result is None:
            continue
        _x, _y, _w, _h, weight = result
        if coarse_best is None or weight > coarse_best[0]:
            coarse_best = (weight, angle)

    if coarse_best is None or coarse_best[0] < MIN_FACE_CONFIDENCE:
        return None

    _weight, angle = coarse_best

    # Refinamiento: repite la detección a resolución completa solo en el
    # ángulo ganador, para un recuadro preciso (el barrido rápido ya nos
    # dijo cuál ángulo probar, así que esto es una sola pasada extra).
    rotated_full = rotate_image(processed_image, angle)
    result = find_face_box(rotated_full)
    if result is None or result[4] < MIN_FACE_CONFIDENCE:
        return None

    x, y, w, h, _weight = result
    rot_h, rot_w = rotated_full.shape[:2]
    return angle, x, y, w, h, rot_w, rot_h


def expand_face_to_portrait(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
):
    """Expande la caja ajustada de la cara a un encuadre cabeza+hombros."""
    x0 = max(0, int(x - w * FACE_MARGIN_LEFT))
    y0 = max(0, int(y - h * FACE_MARGIN_TOP))
    x1 = min(img_w, int(x + w + w * FACE_MARGIN_RIGHT))
    y1 = min(img_h, int(y + h + h * FACE_MARGIN_BOTTOM))
    return x0, y0, x1, y1


# --- Contorno tipo foto-carnet (método de respaldo) --------------------


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

        edge_density = (edges[y : y + h, x : x + w] > 0).mean()
        if edge_density > MAX_EDGE_DENSITY:
            # Patrón de alta frecuencia (QR, código de barras, texto denso):
            # no es una foto de persona.
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
    """
    Detecta la foto de la persona dentro del documento y la recorta. Si el
    documento está girado (foto tomada en ángulo), primero lo orienta antes
    de recortar (ver find_best_face / FACE_SEARCH_ANGLES).
    """
    processed, ratio = resize_for_processing(image)
    img_h, img_w = image.shape[:2]

    face_result = find_best_face(processed)
    photo_box, all_contours = find_photo_region(processed)

    if debug_path is not None:
        if face_result is not None:
            angle, fx, fy, fw, fh, _rot_w, _rot_h = face_result
            debug_img = rotate_image(processed, angle).copy()
            cv2.rectangle(debug_img, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            if angle == 0:
                cv2.drawContours(debug_img, all_contours, -1, (0, 255, 0), 1)
                if photo_box is not None:
                    px, py, pw, ph = photo_box
                    cv2.rectangle(
                        debug_img, (px, py), (px + pw, py + ph), (0, 0, 255), 2
                    )
        else:
            debug_img = processed.copy()
            cv2.drawContours(debug_img, all_contours, -1, (0, 255, 0), 1)
            if photo_box is not None:
                px, py, pw, ph = photo_box
                cv2.rectangle(
                    debug_img, (px, py), (px + pw, py + ph), (0, 0, 255), 2
                )
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_img)

    if face_result is not None:
        angle, x, y, w, h, _rot_w, _rot_h = face_result
        rotated_full = rotate_image(image, angle)
        rfh, rfw = rotated_full.shape[:2]
        x, y, w, h = x * ratio, y * ratio, w * ratio, h * ratio
        x0, y0, x1, y1 = expand_face_to_portrait(x, y, w, h, rfw, rfh)
        return rotated_full[y0:y1, x0:x1]

    if photo_box is not None:
        # Escala el recuadro de vuelta a la resolución original y agrega un
        # pequeño padding para no cortar el borde de la foto.
        x, y, w, h = photo_box
        x, y, w, h = x * ratio, y * ratio, w * ratio, h * ratio
        pad_x, pad_y = w * CROP_PADDING_RATIO, h * CROP_PADDING_RATIO

        x0 = max(0, int(x - pad_x))
        y0 = max(0, int(y - pad_y))
        x1 = min(img_w, int(x + w + pad_x))
        y1 = min(img_h, int(y + h + pad_y))
        return image[y0:y1, x0:x1]

    print(
        "  ! No se detectó una cara ni una foto de credencial en el "
        "documento; se devuelve la imagen completa.",
        file=sys.stderr,
    )
    return image


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
