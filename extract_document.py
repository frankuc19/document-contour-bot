"""
Descarga una imagen desde una URL (p. ej. un link de S3 de un documento tipo
licencia/credencial) y recorta la foto de la persona: busca un rostro en
cualquier tamaño y, si el documento está girado, en las orientaciones
habituales. Si hay un rostro, el recorte sale de esa detección.

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
# Lado máximo de la imagen donde corre el detector. El recorte final se
# hace sobre la imagen original. No se achica más que esto: un rostro
# chico (foto carnet dentro de una toma amplia) tiene que seguir teniendo
# píxeles suficientes.
DETECT_MAX_SIDE = 1280

# Score de YuNet (0-1). Por encima de esto se acepta el rostro y se recorta.
MIN_FACE_SCORE = 0.6

# El documento puede venir derecho, de lado o de cabeza. YuNet encuentra
# el rostro en un rango amplio de tamaño y de inclinación leve; estas
# cuatro orientaciones cubren la foto tomada con el documento girado.
FACE_SEARCH_ANGLES = (0, 90, 180, -90)

# Márgenes para pasar de la caja del rostro (frente, ojos, mentón) a un
# encuadre tipo retrato (cabeza y hombros). Relativos al ancho/alto de
# esa caja.
FACE_MARGIN_LEFT = 0.4
FACE_MARGIN_RIGHT = 0.4
FACE_MARGIN_TOP = 0.45
FACE_MARGIN_BOTTOM = 0.65

_YUNET_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
_face_detector: cv2.FaceDetectorYN | None = None


def download_image(url: str) -> np.ndarray:
    """Descarga una imagen desde una URL y la decodifica con OpenCV (BGR)."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo decodificar la imagen descargada de: {url}")
    return image


def resize_for_processing(image: np.ndarray, max_side: int = DETECT_MAX_SIDE):
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image.copy(), 1.0
    ratio = long_side / float(max_side)
    resized = cv2.resize(image, (int(w / ratio), int(h / ratio)))
    return resized, ratio


def _get_face_detector() -> cv2.FaceDetectorYN:
    global _face_detector
    if _face_detector is None:
        if not _YUNET_PATH.is_file():
            raise FileNotFoundError(
                f"No está el modelo de rostros en {_YUNET_PATH}"
            )
        _face_detector = cv2.FaceDetectorYN.create(
            str(_YUNET_PATH),
            "",
            (320, 320),
            score_threshold=MIN_FACE_SCORE,
            nms_threshold=0.3,
            top_k=100,
        )
    return _face_detector


def _yunet_input_size(width: int, height: int) -> tuple[int, int]:
    """YuNet exige que el lado de entrada sea múltiplo de 32."""
    return ((width + 31) // 32) * 32, ((height + 31) // 32) * 32


def find_face_box(image: np.ndarray):
    """
    Busca el rostro con mayor score en la imagen. Devuelve
    (x, y, w, h, score) o None si no hay ninguno por encima del umbral.
    El tamaño del rostro no importa: el detector corre sobre la imagen
    tal como llega.
    """
    height, width = image.shape[:2]
    input_w, input_h = _yunet_input_size(width, height)
    padded = image
    if (input_w, input_h) != (width, height):
        padded = cv2.copyMakeBorder(
            image,
            0,
            input_h - height,
            0,
            input_w - width,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    detector = _get_face_detector()
    detector.setInputSize((input_w, input_h))
    _, faces = detector.detect(padded)
    if faces is None or len(faces) == 0:
        return None

    best = max(faces, key=lambda face: float(face[-1]))
    x, y, w, h = (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h, float(best[-1])


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


def find_best_face(processed_image: np.ndarray):
    """
    Encuentra el rostro con más score, probando el documento en las
    orientaciones de FACE_SEARCH_ANGLES. Devuelve (angulo, x, y, w, h,
    score) en coordenadas de la imagen rotada ese ángulo, o None.
    """
    best = None  # (score, angle, x, y, w, h)
    for angle in FACE_SEARCH_ANGLES:
        rotated = rotate_image(processed_image, angle)
        result = find_face_box(rotated)
        if result is None:
            continue
        x, y, w, h, score = result
        if best is None or score > best[0]:
            best = (score, angle, x, y, w, h)

    if best is None:
        return None

    score, angle, x, y, w, h = best
    return angle, x, y, w, h, score


def expand_face_to_portrait(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
):
    """Expande la caja del rostro a un encuadre cabeza+hombros."""
    x0 = max(0, int(x - w * FACE_MARGIN_LEFT))
    y0 = max(0, int(y - h * FACE_MARGIN_TOP))
    x1 = min(img_w, int(x + w + w * FACE_MARGIN_RIGHT))
    y1 = min(img_h, int(y + h + h * FACE_MARGIN_BOTTOM))
    return x0, y0, x1, y1


def extract_document(image: np.ndarray, debug_path: Path | None = None) -> np.ndarray:
    """
    Si detecta un rostro, recorta un retrato alrededor. Si el documento
    está girado, lo orienta antes de recortar. Sin rostro, devuelve la
    imagen completa.
    """
    processed, ratio = resize_for_processing(image)
    face_result = find_best_face(processed)

    if debug_path is not None:
        if face_result is not None:
            angle, fx, fy, fw, fh, _score = face_result
            debug_img = rotate_image(processed, angle).copy()
            cv2.rectangle(debug_img, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        else:
            debug_img = processed.copy()
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_img)

    if face_result is None:
        print(
            "  ! No se detectó un rostro; se devuelve la imagen completa.",
            file=sys.stderr,
        )
        return image

    angle, x, y, w, h, _score = face_result
    rotated_full = rotate_image(image, angle)
    rfh, rfw = rotated_full.shape[:2]
    x, y, w, h = x * ratio, y * ratio, w * ratio, h * ratio
    x0, y0, x1, y1 = expand_face_to_portrait(x, y, w, h, rfw, rfh)
    return rotated_full[y0:y1, x0:x1]


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
        print(f"  -> Debug (rostro detectado): {debug_path}")


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
        help="Guarda también una imagen con el rostro detectado dibujado",
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
