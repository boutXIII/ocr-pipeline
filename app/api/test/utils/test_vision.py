from app.api.schemas import DetectionIn, OCRIn, RecognitionIn
from app.api.vision import init_predictor
from onnxtr.models.detection.predictor import DetectionPredictor
from onnxtr.models.predictor import OCRPredictor
from onnxtr.models.recognition.predictor import RecognitionPredictor


def test_vision():
    assert isinstance(init_predictor(OCRIn()), OCRPredictor)
    assert isinstance(init_predictor(DetectionIn()), DetectionPredictor)
    assert isinstance(init_predictor(RecognitionIn()), RecognitionPredictor)