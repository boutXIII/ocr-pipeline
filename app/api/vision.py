import importlib

from collections.abc import Callable

from onnxtr.models import ocr_predictor

from app.api.utils import find_model_path

from .schemas import DetectionIn, OCRIn, RecognitionIn

def init_predictor(request: OCRIn | RecognitionIn | DetectionIn) -> Callable:
    """Initialize the predictor based on the request

    Args:
        request: input request

    Returns:
        Callable: the predictor
    """
    params = request.model_dump()
    params["det_arch"] = params.get("det_arch", "db_resnet50")
    params["reco_arch"] = params.get("reco_arch", "crnn_vgg16_bn")
    # print(f"params = {params}")
    if "det_arch" in params:
        print(f"Loading det model {params['det_arch']}...")
        det_path = find_model_path(params["det_arch"])
        # print(f"det_path = {det_path}")
        det_module = importlib.import_module("onnxtr.models.detection")
        det_model_fn = getattr(det_module, params["det_arch"])
        det_model = det_model_fn(det_path)
        params["det_arch"] = det_model
    if "reco_arch" in params:
        print(f"Loading reco model {params['reco_arch']}...")
        reco_path = find_model_path(params["reco_arch"])
        # print(f"reco_path = {reco_path}")
        reco_module = importlib.import_module("onnxtr.models.recognition")
        reco_model_fn = getattr(reco_module, params["reco_arch"])
        reco_model = reco_model_fn(reco_path)
        params["reco_arch"] = reco_model
    # print(f"params new = {params}")
    bin_thresh = params.pop("bin_thresh", 0.3)
    box_thresh = params.pop("box_thresh", 0.1)
    predictor = ocr_predictor(**params)
    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
    
    if isinstance(request, (OCRIn, RecognitionIn, DetectionIn)):
        if isinstance(request, DetectionIn):
            return predictor.det_predictor
        if isinstance(request, RecognitionIn):
            return predictor.reco_predictor
        if isinstance(request, OCRIn):
            return predictor