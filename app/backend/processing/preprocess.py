# app/backend/processing/preprocess.py
import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Dict, List, Tuple
import torch

from onnxtr.io import DocumentFile  # 🔁 ONNXTR pour PDF/Images
from backend.processing.utils import load_model, bilinear_unwarping, IMG_SIZE


# ---------- Chargement ----------

def _exif_safe_open_image(file_bytes: bytes) -> Image.Image:
    pil = Image.open(io.BytesIO(file_bytes))
    return ImageOps.exif_transpose(pil).convert("RGB")


def load_pages(uploaded_file) -> Tuple[List[np.ndarray], List[str]]:
    """
    Charge un fichier uploadé (image ou PDF) en liste de pages RGB (numpy)
    + liste de types ("pdf" ou "image").
    """
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        doc = DocumentFile.from_pdf(uploaded_file.read())
        return [np.array(p) for p in doc], ["pdf"] * len(doc)
    pil = _exif_safe_open_image(uploaded_file.read())
    return [np.array(pil)], ["image"]


# ---------- Options simples additionnelles ----------

def correct_perspective(img_rgb: np.ndarray, debug: bool = False) -> np.ndarray:

    # Charge dans le modèle
    device = torch.device("cpu")
    model = load_model().to(device)
    model.eval()

    # Load image
    # img = cv2.imread(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0)

    # Make prediction
    inp = inp.to(device)
    point_positions2D, _ = model(inp)

    # Unwarp
    size = img.shape[:2][::-1]
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
        point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
        img_size=tuple(size),
    )
    unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Save result
    unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(unwarped_BGR, cv2.COLOR_BGR2RGB)


def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def binarize_otsu(img_rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)


def binarize_adaptive(img_rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 15)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)


def clean(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge([l, a, b])
    eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return cv2.fastNlMeansDenoisingColored(eq, None, 5, 5, 7, 21)


# ---------- Pipeline configurable ----------

def preprocess_image(img_rgb: np.ndarray, opts: Dict[str, bool]) -> np.ndarray:
    """
    opts:
        correct_perspective: bool
        grayscale: bool
        binarize_otsu: bool
        binarize_adaptive: bool
        clean: bool
    """
    out = img_rgb.copy()

    if opts.get("correct_perspective"):
        out = correct_perspective(out, debug=opts.get("debug", False))

    if opts.get("clean"):
        out = clean(out)

    if opts.get("grayscale"):
        out = to_grayscale(out)

    if opts.get("binarize_adaptive"):
        out = binarize_adaptive(out)
    elif opts.get("binarize_otsu"):
        out = binarize_otsu(out)

    return out
