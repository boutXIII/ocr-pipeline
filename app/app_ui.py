# ===========================================
# 🧠 OCR App UI — docTR ONNX + PaddleOCR ONNX
# ===========================================
# Interface Streamlit unifiée :
# - docTR (via OnnxTR, ONNXRuntime)
# - PaddleOCR (ONNXRuntime)
# ===========================================

import sys, os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from app.logger import get_logger
logger = get_logger("UI")

from onnxtr.utils.visualization import visualize_page

from onnxtr.io import DocumentFile
from onnxtr.contrib.artefacts import ArtefactDetector

from backend.ocr.doctr.onnxtr import (
    DET_ARCHS,
    RECO_ARCHS,
    forward_image,
    load_predictor,
)

from backend.processing.preprocess import load_pages, preprocess_image
from backend.extraction.pipeline import extract_all
from backend.extraction.fields import draw_extracted_fields
from backend.extraction.digit_predict_onnx import process_digits_in_doctr
# from backend.layout.detect import detect_layout

# Ajoute le répertoire contenant app_ui.py au PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# ---------------------------------------------------------
# ⚙️ Encodage console Windows -> UTF-8
# ---------------------------------------------------------
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)
    except Exception:
        pass


# ---------------------------------------------------------
# 🔠 Extraction de texte (docTR export -> texte brut)
# ---------------------------------------------------------
def extract_text_from_page(page_export: dict) -> str:
    logger.debug("extract_text_from_page")
    lines = []
    for block in page_export.get("blocks", []):
        for line in block.get("lines", []):
            words = [w.get("value", "") for w in line.get("words", [])]
            if words:
                lines.append(" ".join(words))
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------
# 🎨 Dessin des zones OCR (docTR export)
# ---------------------------------------------------------
def draw_ocr_boxes(image_bgr: np.ndarray, page_export: dict, color=(0, 255, 255), alpha=0.4) -> np.ndarray:
    logger.debug("draw_ocr_boxes")
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    for block in page_export.get("blocks", []):
        for line in block.get("lines", []):
            for word in line.get("words", []):
                # geometry [[x1,y1],[x2,y2]] normalisée 0..1
                box = np.array(word["geometry"])
                pts = (box * np.array([[w, h]])).astype(int)
                cv2.rectangle(overlay, tuple(pts[0]), tuple(pts[1]), color, thickness=-1)
    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)


# ---------------------------------------------------------
# 🔥 MAIN UI
# ---------------------------------------------------------
def main(det_archs, reco_archs):
    logger.debug("main ui")
    st.set_page_config(layout="wide")
    st.sidebar.title("Moteur OCR")
    engine = st.sidebar.radio(
        "Choisir le moteur OCR :",
        ["docTR (ONNX)"],
    )

    st.sidebar.title("Sélection du document")
    uploaded_file = st.sidebar.file_uploader("Importer une image ou un PDF", type=["pdf", "png", "jpeg", "jpg"])

    pages = []
    origins = []
    page = None
    page_idx = 0

    if uploaded_file is not None:
        pages, origins = load_pages(uploaded_file)
        if not pages:
            st.sidebar.error("Aucune page détectée.")
        else:
            page_idx = st.sidebar.selectbox("Page à analyser :", list(range(1, len(pages) + 1))) - 1
            page = pages[page_idx]  # np.ndarray RGB

    st.sidebar.title("🛠️ Pré-traitement")
    opt_perspective = st.sidebar.checkbox("📐 Correction perspective", value=False)
    opt_clean = st.sidebar.checkbox("🧽 Nettoyage automatique", value=False)
    opt_gray = st.sidebar.checkbox("🌫️ Niveau de gris", value=False)
    opt_otsu = st.sidebar.checkbox("⚫️⚪️ Binarisation OTSU", value=False)
    opt_adapt = st.sidebar.checkbox("🧮 Binarisation adaptative", value=False)
    apply_preprocess = st.sidebar.button("✅ Appliquer pré-traitement")

    pre_opts = {
        "correct_perspective": opt_perspective,
        "clean": opt_clean,
        "grayscale": opt_gray,
        "binarize_otsu": opt_otsu,
        "binarize_adaptive": opt_adapt,
    }

    # Colonnes d'affichage
    if engine.startswith("docTR"):
        cols = st.columns(5)
        cols[0].subheader("📄 Page d’entrée")
        cols[1].subheader("🛠️ Pré-traitement")
        cols[2].subheader("🟠 Segmentation")
        cols[3].subheader("🔠 Zones OCR")
        cols[4].subheader("📘 Reconstitution")
    else:
        cols = st.columns((1, 1, 1))
        cols[0].subheader("📄 Page d’entrée")
        cols[1].subheader("🛠️ Pré-traitement")
        cols[2].subheader("🧾 Résultats")

    processed = None
    if page is not None:
        cols[0].image(page, caption=f"Page {page_idx+1} ({origins[page_idx]})", width="stretch")

        if apply_preprocess:
            st.session_state["processed_image"] = preprocess_image(page, pre_opts)

        processed = st.session_state.get("processed_image", page)
        cols[1].image(processed, caption="(Pré-traitée)", width="stretch")

    st.sidebar.divider()

    # === Panneau de config par moteur ===
    if engine.startswith("docTR"):
        st.sidebar.title("⚙️ Modèles docTR (ONNX)")
        det_arch = st.sidebar.selectbox(
            "Modèle de détection",
            det_archs,
            index=det_archs.index("db_resnet50") if "db_resnet50" in det_archs else 0,
        )
        reco_arch = st.sidebar.selectbox(
            "Modèle de reconnaissance",
            reco_archs,
            index=reco_archs.index("crnn_vgg16_bn") if "crnn_vgg16_bn" in reco_archs else 0,
        )

        st.sidebar.divider()
        detect_language = st.sidebar.checkbox("Détection de la langue", value=True)
        assume_straight_pages = st.sidebar.checkbox("Supposer les pages droites", value=False)
        disable_page_orientation = st.sidebar.checkbox("Désactiver orientation page", value=False)
        disable_crop_orientation = st.sidebar.checkbox("Désactiver orientation découpe", value=False)
        straighten_pages = st.sidebar.checkbox("Redresser les pages", value=True)
        export_straight_boxes = st.sidebar.checkbox("Exporter zones droites", value=True)
        bin_thresh = st.sidebar.slider("Seuil de binarisation", 0.1, 0.9, 0.3, 0.1)
        box_thresh = st.sidebar.slider("Seuil de détection de boîte", 0.1, 0.9, 0.1, 0.1)

    else:
        st.sidebar.title("⚙️ PaddleOCR Configuration (ONNX)")
        lang = st.sidebar.selectbox("Langue OCR", ["fr", "en"], index=0)

    if st.sidebar.button("Analyser la page"):
        if uploaded_file is None or page is None:
            st.sidebar.error("Veuillez importer un document avant l’analyse.")
            return

        input_img = processed if processed is not None else page

        # -------------------------------
        # 🔹 Mode docTR (ONNX)
        # -------------------------------
        if engine.startswith("docTR"):
            with st.spinner("Chargement du modèle docTR (ONNX)..."):
                predictor = load_predictor(
                    det_arch=det_arch,
                    reco_arch=reco_arch,
                    assume_straight_pages=assume_straight_pages,
                    straighten_pages=straighten_pages,
                    export_as_straight_boxes=export_straight_boxes,
                    disable_page_orientation=disable_page_orientation,
                    disable_crop_orientation=disable_crop_orientation,
                    detect_language=detect_language,
                    bin_thresh=bin_thresh,
                    box_thresh=box_thresh,
                    device=None,
                )

            with st.spinner("Analyse docTR (ONNX) en cours..."):
                start_time = time.perf_counter()
                seg_map = forward_image(predictor, input_img)
                seg_map = np.squeeze(seg_map)
                seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis("off")
                cols[2].pyplot(fig)

                out = predictor([input_img])
                fig = visualize_page(out.pages[0].export(), out.pages[0].page, interactive=False, add_labels=False)
                cols[3].pyplot(fig)

                page_export = out.pages[0].export()
                import copy
                page_export_before_digits = copy.deepcopy(page_export)

                # 🔢 Correction chiffres manuscrits (ONNX)
                try:
                    page_export = process_digits_in_doctr(page_export, input_img)
                except Exception as e:
                    print("Digit ONNX error:", e)

                if assume_straight_pages or straighten_pages:
                    try:
                        img = out.pages[0].synthesize()
                        cols[4].image(img, clamp=True, width="stretch")
                    except Exception:
                        pass

                elapsed_docTR = time.perf_counter() - start_time
                st.success(f"✅ OCR docTR (ONNX) terminé en {elapsed_docTR:.2f} s")

                tabs = st.tabs(["👁️ Visualisation", "📦 Extraction JSON", "🧩 Layout / Structure", "🔢 Digits"])

                with tabs[0]:
                    zone_col, text_col = st.columns(2)

                    with zone_col:
                        zone_col.subheader("🟡 Zones OCR")
                        proc_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                        highlighted = draw_ocr_boxes(proc_bgr, page_export)
                        zone_col.image(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB), width="stretch")

                    with text_col:
                        text_col.subheader("🧾 Texte reconnu")
                        text_content = extract_text_from_page(page_export)
                        line_count = text_content.count("\n") + 1
                        dynamic_height = min(900, max(250, line_count * 20))
                        text_col.text_area("🧾 Texte reconnu", text_content, height=dynamic_height)

                with tabs[1]:
                    zone_col, text_col = st.columns(2)

                    with zone_col:
                        zone_col.subheader("🟢 Champs extraits")
                        text_content = extract_text_from_page(page_export)
                        extracted = extract_all(text_content)
                        drawn_img = draw_extracted_fields(
                            cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR),
                            page_export,
                            extracted,
                        )
                        from backend.extraction.gliner.draw_gliner import draw_gliner_entities
                        gliner_entities = extracted.get("gliner_entities", [])
                        drawn_img = draw_gliner_entities(drawn_img, gliner_entities, page_export)
                        zone_col.image(cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB), width="stretch")

                    with text_col:
                        text_col.subheader("📦 Export JSON docTR (ONNX)")
                        text_content = extract_text_from_page(page_export)
                        extracted = extract_all(text_content)
                        st.json(extracted)

                with tabs[2]:
                    st.subheader("📐 Détection de structure du document (Layout)")
                    doc = DocumentFile.from_images(uploaded_file)
                    detector = ArtefactDetector(labels=["bar_code", "qr_code", "logo", "photo"])
                    artefacts = detector(doc)
                    logger.debug("artefacts %s", artefacts)

                with tabs[3]:
                    st.subheader("🔢 Comparaison de lecture des chiffres manuscrits")

                    # ----- Texte avant correction -----
                    before_text = extract_text_from_page(page_export_before_digits)

                    # ----- Texte après correction -----
                    after_text = extract_text_from_page(page_export)

                    col_before, col_after = st.columns(2)

                    # =============================
                    # 📄 Colonne : Avant correction
                    # =============================
                    with col_before:
                        st.caption("🟡 AVANT correction (docTR)")
                        line_count = before_text.count("\n") + 1
                        height = min(900, max(200, line_count * 20))
                        st.text_area("DocTR original", before_text, height=height)

                    # =============================
                    # 📄 Colonne : Après correction
                    # =============================
                    with col_after:
                        st.caption("🟢 APRÈS correction (Digits ONNX)")
                        line_count = after_text.count("\n") + 1
                        height = min(900, max(200, line_count * 20))
                        st.text_area("Digits corrigés", after_text, height=height)

                    # =============================
                    # 🔍 Diff visuelle
                    # =============================
                    st.subheader("🆚 Diff (chiffres modifiés en vert)")

                    before_lines = before_text.split("\n")
                    after_lines = after_text.split("\n")

                    diff_html = []

                    for b_line, a_line in zip(before_lines, after_lines):

                        if b_line == a_line:
                            diff_html.append(a_line)
                            continue

                        # Construction ligne modifiée
                        line_out = ""
                        min_len = min(len(b_line), len(a_line))

                        for i in range(min_len):
                            if b_line[i] != a_line[i]:
                                # highlight les différences
                                line_out += f"<span style='color:#00FF00;font-weight:bold;'>{a_line[i]}</span>"
                            else:
                                line_out += a_line[i]

                        # Si la ligne "après" est plus longue
                        if len(a_line) > min_len:
                            extra = a_line[min_len:]
                            line_out += f"<span style='color:#00FF00;font-weight:bold;'>{extra}</span>"

                        diff_html.append(line_out)

                    st.markdown("<br>".join(diff_html), unsafe_allow_html=True)
if __name__ == "__main__":
    main(DET_ARCHS, RECO_ARCHS)
