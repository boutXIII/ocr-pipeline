def draw_extracted_fields(img_rgb, page_export, extracted_fields):
    """
    Dessine uniquement les zones liées aux champs extraits :
    - On cherche les mots correspondant aux valeurs extraites.
    - On encadre uniquement ces mots.
    """
    import cv2
    import numpy as np

    img = img_rgb.copy()
    h, w = img.shape[:2]

    # ---- FLATTEN + nettoyer ----
    values = []

    for v in extracted_fields.values():
        if isinstance(v, list):
            # Liste de dates ou liste de dict -> on extrait intelligemment
            for item in v:
                if isinstance(item, dict) and "date" in item:
                    values.append(item["date"])        # on garde la date
                elif isinstance(item, str):
                    values.append(item)
        elif isinstance(v, str):
            values.append(v)

    # enlever None + vider + normaliser espaces + minuscules
    values = [v.lower().strip() for v in values if isinstance(v, str) and v.strip()]

    if not values:
        return img  # rien à encadrer

    # ---- Recherche et dessin ----
    for block in page_export.get("blocks", []):
        for line in block.get("lines", []):
            for word in line.get("words", []):
                txt = word["value"].lower().strip()

                # match "soft" : doit contenir ou être contenu
                if any(v in txt or txt in v for v in values):
                    (x1, y1), (x2, y2) = word["geometry"]
                    x1, y1 = int(x1 * w), int(y1 * h)
                    x2, y2 = int(x2 * w), int(y2 * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 3)

    return img
