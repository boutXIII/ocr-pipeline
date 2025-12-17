import os
from typing import Any, Optional

import base64
import cv2
import json
import numpy as np
from fastapi import UploadFile

from onnxtr.io import DocumentFile

from gliner import GLiNER
from app.api.ner_config import DOCUMENT_CLASSES

DEFAULT_ONNXTR_CACHE = "app\models\doctr"
os.environ["ONNXTR_CACHE_DIR"] = DEFAULT_ONNXTR_CACHE

def resolve_geometry(
    geom: Any,
) -> tuple[float, float, float, float] | tuple[float, float, float, float, float, float, float, float]:
    if len(geom) == 4:
        return (*geom[0], *geom[1], *geom[2], *geom[3])
    return (*geom[0], *geom[1])


async def get_documents(
    request,
    file: Optional[UploadFile] = None,
) -> tuple[DocumentFile, str]:
    """Convert a list of UploadFile objects to lists of numpy arrays and their corresponding filenames
    Support:
    - UploadFile (multipart)
    - body brut (image/pdf)
    - JSON base64

    Args:
        request: Request object containing the files to be processed

    Returns:
    - DocumentFile
    - filename

    """

    filename = "document"

    if file is not None:
        content = await file.read()
        filename = file.filename or filename

    else:
        content = await request.body()

        if not content:
            raise ValueError("Empty body")

        # Tentative JSON base64
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and "fileBase64" in payload:
                content = base64.b64decode(payload["fileBase64"])
                filename = payload.get("filename", filename)
        except Exception:
            pass  # body binaire brut

    try:
        if (
            filename.lower().endswith(".pdf")
            or content[:4] == b"%PDF"
        ):
            doc = DocumentFile.from_pdf(content)
        else:
            doc = DocumentFile.from_images(content)
    except Exception as e:
        raise ValueError(f"Error loading document: {e}")

    return doc, filename

# -------------------------------------------------------------------
# 🔍 Recherche automatique des fichiers .pt selon le modèle choisi
# -------------------------------------------------------------------
def find_model_path(model_name: str, models_dir: str = DEFAULT_ONNXTR_CACHE) -> str:
    """
    Recherche un fichier .pt correspondant au modèle choisi dans le cache local.
    Exemple : det_arch='db_resnet50' → 'db_resnet50-xxxx.pt'
    """
    # logger.debug("find_model_path")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Le dossier '{models_dir}' n'existe pas.")
    
    for file in os.listdir(models_dir):
        if file.startswith(model_name) and file.endswith(".onnx"):
            return os.path.join(models_dir, file)
    raise FileNotFoundError(f"Aucun fichier trouvé pour {model_name} dans {models_dir}")

# charge UNE fois
gliner_model = GLiNER.from_pretrained(
    "app/models/gliner/gliner_large-v2.5",
    load_onnx_model=True,
    # multi_label=True,
    map_location="cpu",
)

def run_gliner(
    text: str,
    document_class: str,
    threshold: float,
):
    labels = DOCUMENT_CLASSES.get(document_class)

    if not labels:
        raise ValueError(f"Classe de document inconnue: {document_class}")

    return gliner_model.predict_entities(
        text,
        labels=labels,
        threshold=threshold,
    )

def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

def stroke_variance(bin_img):
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(bin_img, kernel)
    eroded = cv2.erode(bin_img, kernel)
    diff = cv2.absdiff(dilated, eroded)
    return np.var(diff)

def line_regularity(bin_img):
    projection = np.sum(bin_img, axis=1)
    peaks = np.where(projection > np.mean(projection))[0]
    if len(peaks) < 2:
        return 0
    diffs = np.diff(peaks)
    return np.std(diffs)

def glyph_repeatability(bin_img):
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    widths = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if h > 10 and w > 5:
            widths.append((w,h))

    if len(widths) < 20:
        return 0

    sizes = np.array(widths)
    return np.std(sizes[:,0]) + np.std(sizes[:,1])

def print_type(img):
    bin_img = binarize(img)

    stroke = stroke_variance(bin_img)
    line = line_regularity(bin_img)
    glyph = glyph_repeatability(bin_img)

    printed_score = 0
    handwritten_score = 0

    if stroke < 50: printed_score += 1
    else: handwritten_score += 1

    if line < 20: printed_score += 1
    else: handwritten_score += 1

    if glyph < 15: printed_score += 1
    else: handwritten_score += 1

    if printed_score == 3:
        return "dactylographié"
    elif handwritten_score == 3:
        return "manuscrit"
    else:
        return "mixte"
