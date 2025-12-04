# ===========================================
# 🧠 OCR App API (FastAPI) — docTR ONNX
# ===========================================
# - Utilise FastAPI natif pour l'OpenAPI
# - Compatible avec ton schéma MSP existant
# - Endpoints : /health, /extract/ocr
# ===========================================

import os
import io
import sys
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, EngineConfig
from app.backend.ocr.doctr.onnxtr import load_predictor

# --- UTF-8 stdout ---
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# --- États globaux ---
DEVICE = "onnxruntime-cpu"

app_state = {
    "predictor": None,
    "models_loaded": False,
    "engineType": "Doctr-ONNX",
}

# -----------------------------------------------------
# ⚙️ Charger modèle docTR ONNX (OnnxTR)
# -----------------------------------------------------
def load_doctr():
    print("🔧 Chargement du modèle DocTR ONNX (db_resnet50 + crnn_vgg16_bn)")
    # engine_cfg = EngineConfig(
    #     providers=[("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
    # )

    predictor = load_predictor(
        det_arch="db_resnet50",
        reco_arch="crnn_vgg16_bn",
        assume_straight_pages=True,
        straighten_pages=True,
        export_as_straight_boxes=True,
        disable_page_orientation=False,
        disable_crop_orientation=False,
        detect_language=True,
        bin_thresh=0.3,
        box_thresh=0.1,
        device=None,
    )

    # predictor = ocr_predictor(
    #     det_arch="db_resnet50",
    #     reco_arch="crnn_vgg16_bn",
    #     assume_straight_pages=True,
    #     straighten_pages=True,
    #     export_as_straight_boxes=True,
    #     detect_orientation=False,
    #     disable_page_orientation=False,
    #     disable_crop_orientation=False,
    #     detect_language=True,
    #     det_engine_cfg=engine_cfg,
    #     reco_engine_cfg=engine_cfg,
    # )

    app_state["predictor"] = predictor
    app_state["models_loaded"] = True
    return predictor


# -----------------------------------------------------
# 🌱 Lifecycle (startup / shutdown)
# -----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Au démarrage : précharger le modèle
    load_doctr()
    yield
    # À l'arrêt : on pourrait libérer des ressources ici


app = FastAPI(
    title="OCR App API (docTR ONNX)",
    description="API OCR basée sur docTR via OnnxTR (ONNXRuntime)",
    version="0.0.1",
    lifespan=lifespan,
)


# -----------------------------------------------------
# 🩺 /health
# -----------------------------------------------------
@app.get(
    "/health",
    tags=["health"],
    summary="Health check endpoint",
    description="Return API and model status",
    response_description="200 OK — Model and API status",
)
async def health_check():
    return {
        "status": "ok",
        "engine": app_state["engineType"],
        "models_loaded": app_state["models_loaded"],
        "device": str(DEVICE),
    }


# -----------------------------------------------------
# 📥 /extract/ocr
# -----------------------------------------------------
@app.post(
    "/extract/ocr",
    tags=["extract"],
    summary="Extract text from the given image and organize it into Block -> Line -> Word",
    description=(
        "OCR Endpoint using docTR (ONNX) via OnnxTR.\n\n"
        "You can send file either :\n"
        "1️⃣ As a binary upload (e.g. `Content-Type: image/jpeg`)\n"
        "2️⃣ As JSON `{ \"fileBase64\": \"...\" }`"
    ),
    response_description="Structured OCR result",
)
async def extract_ocr(
    request: Request,
    file: UploadFile = File(
        None,
        description="Upload image or PDF file (binary)",
    ),
    engineType: str = Query(
        "Doctr-ONNX",
        description="OCR engine to use (currently docTR-ONNX only)",
    ),
    imageEnhancement: bool = Query(
        False,
        description="Apply image preprocessing (deskew, despeckle, autorotate)",
    ),
    binarize: bool = Query(
        False,
        description="Convert image to black/white before processing",
    ),
    absoluteCoord: bool = Query(
        False,
        description="Return absolute coordinate positions",
    ),
):
    """OCR Endpoint (docTR ONNX)

    - Accepts binary uploads or Base64 encoded JSON
    - Returns a list of recognized text blocks and words
    """
    try:
        if file is not None:
            content = await file.read()
        else:
            content = await request.body()

        if not content:
            return JSONResponse(
                status_code=204,
                content={
                    "systemMessage": "Empty body",
                    "humanMessage": "Please provide an image or PDF file.",
                    "messageCode": "EMPTY_BODY",
                },
            )

        # Lecture du document via OnnxTR.DocumentFile
        try:
            if file and file.filename.lower().endswith(".pdf"):
                doc = DocumentFile.from_pdf(content)
            else:
                doc = DocumentFile.from_images(content)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "systemMessage": str(e),
                    "humanMessage": "Error loading document",
                    "messageCode": "LOAD_ERROR",
                },
            )

        predictor = app_state["predictor"]
        result = predictor

        pages = []

        for page in result.pages:
            blocks = []
            for block in page.blocks:
                lines = []
                for line in block.lines:
                    words = []
                    for word in line.words:
                        words.append({
                            "top": word.geometry[0][1],
                            "left": word.geometry[0][0],
                            "right": word.geometry[1][0],
                            "bottom": word.geometry[1][1],
                            "value": word.value,
                            "confidence": float(word.confidence),
                        })

                    # --- Line Text & Confidence ---
                    line_text = " ".join(w["value"] for w in words)
                    line_conf = float(np.mean([w["confidence"] for w in words])) if words else 0.0

                    lines.append({
                        "top": line.geometry[0][1],
                        "left": line.geometry[0][0],
                        "right": line.geometry[1][0],
                        "bottom": line.geometry[1][1],
                        "value": line_text,
                        "confidence": line_conf,
                        "words": words,
                    })

                # --- Block Text & Confidence ---
                block_text = "\n".join(l["value"] for l in lines)
                block_conf = float(np.mean([l["confidence"] for l in lines])) if lines else 0.0

                blocks.append({
                    "top": block.geometry[0][1],
                    "left": block.geometry[0][0],
                    "right": block.geometry[1][0],
                    "bottom": block.geometry[1][1],
                    "value": block_text,
                    "confidence": block_conf,
                    "lines": lines,
                })

            pages.append({
                "pageId": page.page_idx + 1,
                "pageHeight": page.dimensions[1],
                "pageWidth": page.dimensions[0],
                "engine": engineType,
                "engineMode": "accurate",
                "rotationAngle": 0,
                "blocks": blocks,
                "partialReadZone": {"top": 0, "left": 0, "right": 0, "bottom": 0},
            })

        return {
            "status": "success",
            "engine": engineType,
            "pages": pages,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "systemMessage": str(e),
                "humanMessage": "Unexpected processing error",
                "messageCode": "INTERNAL_ERROR",
            },
        )

# -----------------------------------------------------
# 🩺 /extract/read
# -----------------------------------------------------
@app.post(
    "/read",
    tags=["read"],
    summary="Perform OCR on an uploaded file",
    description=(
            "OCR Endpoint using docTR (ONNX) via OnnxTR.\n\n"
            "You can send file either :\n"
            "1️⃣ As a binary upload (e.g. `Content-Type: image/jpeg`)\n"
            "2️⃣ As JSON `{ \"fileBase64\": \"...\" }`"
    ),
    response_description="Structured OCR result",
)
async def extract_ocr(
        request: Request,
        file: UploadFile = File(
            None,
            description="Upload image or PDF file (binary)",
        ),
        engineType: str = Query(
            "Doctr-ONNX",
            description="OCR engine to use (currently docTR-ONNX only)",
        ),
        imageEnhancement: bool = Query(
            False,
            description="Apply image preprocessing (deskew, despeckle, autorotate)",
        ),
        binarize: bool = Query(
            False,
            description="Convert image to black/white before processing",
        ),
        absoluteCoord: bool = Query(
            False,
            description="Return absolute coordinate positions",
        ),
        documentClass: str = Query(
            "SAN_FACTURE_MEDECINE_DOUCE",
            description="Classe de document de lecture",
        ),

):
    """OCR Endpoint (docTR ONNX)

    - Accepts binary uploads or Base64 encoded JSON
    - Returns a list of recognized text blocks and words
    """
    try:
        if file is not None:
            content = await file.read()
        else:
            content = await request.body()

        if not content:
            return JSONResponse(
                status_code=204,
                content={
                    "systemMessage": "Empty body",
                    "humanMessage": "Please provide an image or PDF file.",
                    "messageCode": "EMPTY_BODY",
                },
            )

        # Lecture du document via OnnxTR.DocumentFile
        try:
            if file and file.filename.lower().endswith(".pdf"):
                doc = DocumentFile.from_pdf(content)
            else:
                doc = DocumentFile.from_images(content)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "systemMessage": str(e),
                    "humanMessage": "Error loading document",
                    "messageCode": "LOAD_ERROR",
                },
            )

        predictor = app_state["predictor"]
        result = predictor

        pages = []

        for page in result.pages:
            blocks = []
            for block in page.blocks:
                lines = []
                for line in block.lines:
                    words = []
                    for word in line.words:
                        words.append({
                            "top": word.geometry[0][1],
                            "left": word.geometry[0][0],
                            "right": word.geometry[1][0],
                            "bottom": word.geometry[1][1],
                            "value": word.value,
                            "confidence": float(word.confidence),
                        })

                    # --- Line Text & Confidence ---
                    line_text = " ".join(w["value"] for w in words)
                    line_conf = float(np.mean([w["confidence"] for w in words])) if words else 0.0

                    lines.append({
                        "top": line.geometry[0][1],
                        "left": line.geometry[0][0],
                        "right": line.geometry[1][0],
                        "bottom": line.geometry[1][1],
                        "value": line_text,
                        "confidence": line_conf,
                        "words": words,
                    })

                # --- Block Text & Confidence ---
                block_text = "\n".join(l["value"] for l in lines)
                block_conf = float(np.mean([l["confidence"] for l in lines])) if lines else 0.0

                blocks.append({
                    "top": block.geometry[0][1],
                    "left": block.geometry[0][0],
                    "right": block.geometry[1][0],
                    "bottom": block.geometry[1][1],
                    "value": block_text,
                    "confidence": block_conf,
                    "lines": lines,
                })

            pages.append({
                "pageId": page.page_idx + 1,
                "pageHeight": page.dimensions[1],
                "pageWidth": page.dimensions[0],
                "engine": engineType,
                "engineMode": "accurate",
                "rotationAngle": 0,
                "blocks": blocks,
                "partialReadZone": {"top": 0, "left": 0, "right": 0, "bottom": 0},
            })

        return {
            "status": "success",
            "engine": engineType,
            "pages": pages,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "systemMessage": str(e),
                "humanMessage": "Unexpected processing error",
                "messageCode": "INTERNAL_ERROR",
            },
        )


# -----------------------------------------------------
# ▶️ Lancement direct
# -----------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("🚀 API OCR ONNX démarrée sur http://localhost:8080/docs")
    uvicorn.run(app, host="0.0.0.0", port=8080)
