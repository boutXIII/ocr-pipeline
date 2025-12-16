import os
from typing import Any, Optional

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


async def get_documents(files: list[UploadFile]) -> tuple[list[np.ndarray], list[str]]:  # pragma: no cover
    """Convert a list of UploadFile objects to lists of numpy arrays and their corresponding filenames

    Args:
        files: list of UploadFile objects

    Returns:
        tuple[list[np.ndarray], list[str]]: list of numpy arrays and their corresponding filenames

    """
    filenames = []
    docs = []
    for file in files:
        mime_type = file.content_type
        if mime_type in ["image/jpeg", "image/png"]:
            docs.extend(DocumentFile.from_images([await file.read()]))
            filenames.append(file.filename or "")
        elif mime_type == "application/pdf":
            pdf_content = DocumentFile.from_pdf(await file.read())
            docs.extend(pdf_content)
            filenames.extend([file.filename] * len(pdf_content) or [""] * len(pdf_content))
        else:
            raise ValueError(f"Unsupported file format: {mime_type} for file {file.filename}")

    return docs, filenames

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