# backend/ocr/doctr/onnxtr.py
# ============================================================
# 🧠 docTR ONNX backend (OnnxTR + ONNXRuntime)
# ============================================================
# Ce module remplace le backend PyTorch :
#  - même API que backend/ocr/doctr/pytorch.py
#  - DET_ARCHS / RECO_ARCHS identiques
#  - load_predictor(...) & forward_image(...) compatibles
#  - utilise onnxtr.ocr_predictor (ONNXRuntime CPU)
# ============================================================

import os
import importlib
from pathlib import Path
from typing import Optional

from app.logger import get_logger
logger = get_logger("ONNXTR")

import numpy as np

from onnxtr.models import EngineConfig, ocr_predictor
from onnxtr.models.predictor import OCRPredictor

# ------------------------------------------------------------
# 📦 Config cache ONNXTR
# ------------------------------------------------------------
# Par défaut, on met le cache dans app/models/doctr
BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_ONNXTR_CACHE = BASE_DIR / "models" / "doctr"

os.environ.setdefault("ONNXTR_CACHE_DIR", str(DEFAULT_ONNXTR_CACHE))
DEFAULT_ONNXTR_CACHE.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 📚 Listes d'architectures docTR supportées par OnnxTR
# ------------------------------------------------------------
DET_ARCHS = [
    "fast_base",
    "fast_small",
    "fast_tiny",
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]

RECO_ARCHS = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
    "viptr_tiny",
]

# -------------------------------------------------------------------
# 🔍 Recherche automatique des fichiers .pt selon le modèle choisi
# -------------------------------------------------------------------
def find_model_path(model_name: str, models_dir: str = DEFAULT_ONNXTR_CACHE) -> str:
    """
    Recherche un fichier .pt correspondant au modèle choisi dans le cache local.
    Exemple : det_arch='db_resnet50' → 'db_resnet50-xxxx.pt'
    """
    logger.debug("find_model_path")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Le dossier '{models_dir}' n'existe pas.")
    
    for file in os.listdir(models_dir):
        if file.startswith(model_name) and file.endswith(".onnx"):
            return os.path.join(models_dir, file)
    raise FileNotFoundError(f"Aucun fichier trouvé pour {model_name} dans {models_dir}")


def load_predictor(
    det_arch: str,
    reco_arch: str,
    assume_straight_pages: bool,
    straighten_pages: bool,
    export_as_straight_boxes: bool,
    disable_page_orientation: bool,
    disable_crop_orientation: bool,
    detect_language: bool,
    bin_thresh: float,
    box_thresh: float,
    device: Optional[object] = None,  # ignoré, gardé pour compatibilité
) -> OCRPredictor:
    """Charge un predictor docTR en ONNX via OnnxTR.

    Signature compatible avec backend/ocr/doctr/pytorch.py pour
    pouvoir simplement changer l'import dans le reste du projet.

    Args:
        det_arch: architecture de détection (nom string)
        reco_arch: architecture de reconnaissance (nom string)
        assume_straight_pages: supposer les pages droites
        straighten_pages: redresser les pages
        export_as_straight_boxes: exporter des boîtes droites
        disable_page_orientation: désactiver la détection d’orientation de page
        disable_crop_orientation: désactiver la détection d’orientation de crop
        detect_language: activer la détection de langue
        bin_thresh: seuil de binarisation de la carte de segmentation
        box_thresh: seuil minimal de détection de boîte
        device: ignoré (compatibilité avec backend PyTorch)

    Returns:
        OCRPredictor (OnnxTR)
    """
    logger.debug("load_predictor")
    # Config ONNXRuntime (CPU)
    engine_cfg = EngineConfig(
        providers=[
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
        ]
    )

    det_path = find_model_path(det_arch)
    reco_path = find_model_path(reco_arch)

    det_module = importlib.import_module("onnxtr.models.detection")
    reco_module = importlib.import_module("onnxtr.models.recognition")

    det_model_fn = getattr(det_module, det_arch)
    reco_model_fn = getattr(reco_module, reco_arch)
    logger.debug("Loading detection model from %s", det_model_fn)
    logger.debug("Loading detection model from %s", reco_model_fn)

    det_model = det_model_fn(det_path)
    # det_model = det_model_fn(pretrained=False, pretrained_backbone=False)
    # det_model.from_pretrained(det_path, map_location="cpu")

    reco_model = reco_model_fn(reco_path)
    # reco_model = reco_model_fn(pretrained=False, pretrained_backbone=False)
    # reco_model.from_pretrained(reco_path, map_location="cpu")

    predictor = ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=not assume_straight_pages,
        disable_page_orientation=disable_page_orientation,
        disable_crop_orientation=disable_crop_orientation,
        detect_language=detect_language,
        det_engine_cfg=engine_cfg,
        reco_engine_cfg=engine_cfg,
        clf_engine_cfg=engine_cfg,
    )

    # Applique les seuils de post-processing comme dans la version PyTorch
    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    predictor.det_predictor.model.postprocessor.box_thresh = box_thresh

    return predictor


def forward_image(
    predictor: OCRPredictor,
    image: np.ndarray,
    device: Optional[object] = None,  # ignoré, pour compat avec ancien code
) -> np.ndarray:
    """Passe une image dans le modèle de détection ONNX (docTR).

    Args:
        predictor: instance OCRPredictor (OnnxTR)
        image: image RGB (numpy) à traiter
        device: ignoré (compatibilité avec version PyTorch)

    Returns:
        Carte de segmentation (numpy ndarray)
    """
    logger.debug("forward_image")
    processed_batches = predictor.det_predictor.pre_processor([image])
    out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
    seg_map = out["out_map"]  # OnnxTR renvoie déjà du numpy
    return seg_map
