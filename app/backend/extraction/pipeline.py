# app/backend/extraction/pipeline.py

from backend.extraction.gliner.gliner_onnx import ner_gliner
from backend.extraction.gliner.draw_gliner import draw_gliner_entities
# from .fields import extract_fields
# from .semantic_fields import (
#     extract_patient_and_practitioner,
#     extract_amounts,
#     extract_dates_with_context,
# )

def extract_all(text: str):
    """
    Pipeline d'extraction sémantique à partir de texte brut OCR.
    """
    result = {}
    # result = extract_fields(text)
    # result.update(extract_patient_and_practitioner(text))
    # result.update(extract_amounts(text))
    # result["dates_context"] = extract_dates_with_context(text)

    # 🔥 AJOUT GLINER — extraction NER après docTR
    result["entities_gliner"] = ner_gliner(text)

    return result

