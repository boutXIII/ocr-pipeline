from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

ENGINE_TYPE = "onnxtr-doctr"
DEVICE = "CPU (ONNXRuntime)"

def check_model(name: str) -> bool:
    for root, _, files in os.walk(MODELS_DIR):
        if name in root and "model.onnx" in files:
            return True
    return False


app_state = {
    "engineType": ENGINE_TYPE,
    "models_available": {
        "detection": {
            "db_resnet50": check_model("db_resnet50"),
            "fast_base": check_model("fast_base"),
        },
        "recognition": {
            "crnn_vgg16_bn": check_model("crnn_vgg16_bn"),
            "sar_resnet31": check_model("sar_resnet31"),
        },
    },
    "device": DEVICE,
}
