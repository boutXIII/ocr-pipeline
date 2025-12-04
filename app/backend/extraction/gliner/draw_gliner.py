import cv2
import numpy as np

# Palette standard GLiNER inspirée HuggingFace
PALETTE = {
    "PERSON": (255, 102, 102),
    "ORGANIZATION": (255, 178, 102),
    "LOCATION": (102, 204, 255),
    "DATE": (153, 255, 153),
    "MEDICATION": (255, 153, 255),
    "AMOUNT": (255, 255, 153),
    "PATIENT": (204, 153, 255),
    "DOCTOR": (255, 204, 153),
    "HOSPITAL": (153, 204, 255),
}

DEFAULT_COLOR = (200, 200, 200)  # gris clair


def draw_gliner_entities(image_bgr, entities, page_export):
    """
    Dessine les entités GLiNER directement sur l’image OCRée.
    On se base sur la segmentation docTR pour récupérer les positions exactes des mots.
    
    entities = [
      {"label": "PERSON", "word": "...", "start": ..., "end": ...}
    ]

    page_export = out.pages[0].export()
    """

    img = image_bgr.copy()
    h, w = img.shape[:2]

    # On récupère toutes les boxes docTR pour pouvoir faire correspondance
    boxes = []
    for block in page_export["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                word_text = word["value"]
                (x0, y0), (x1, y1) = word["geometry"]
                boxes.append({
                    "text": word_text,
                    "x0": int(x0 * w),
                    "y0": int(y0 * h),
                    "x1": int(x1 * w),
                    "y1": int(y1 * h),
                })

    # Parcours des entités GLiNER
    for ent in entities:
        label = ent["label"].upper()
        content = ent["text"]

        # couleur choisie
        color = PALETTE.get(label, DEFAULT_COLOR)

        # On cherche le mot correspondant dans les boxes docTR
        for b in boxes:
            # Matching simple : même mot
            if b["text"].lower() == content.lower():
                cv2.rectangle(img, (b["x0"], b["y0"]), (b["x1"], b["y1"]), color, 2)
                cv2.putText(img, label, (b["x0"], b["y0"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                break

    return img
