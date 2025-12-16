from typing import Any, Dict, List

from pydantic import BaseModel, Field

# =========================
# INPUT SCHEMAS
# =========================
class OCRIn(BaseModel):
    resolve_lines: bool = Field(default=True, examples=[True])
    resolve_blocks: bool = Field(default=False, examples=[False])
    paragraph_break: float = Field(default=0.0035, examples=[0.0035])

class RecognitionIn(BaseModel):
    reco_arch: str = Field(default="crnn_vgg16_bn", examples=["crnn_vgg16_bn"])
    reco_bs: int = Field(default=128, examples=[128])

class DetectionIn(BaseModel):
    det_arch: str = Field(default="db_resnet50", examples=["db_resnet50"])
    assume_straight_pages: bool = Field(default=True, examples=[True])
    preserve_aspect_ratio: bool = Field(default=True, examples=[True])
    symmetric_pad: bool = Field(default=True, examples=[True])
    det_bs: int = Field(default=2, examples=[2])
    bin_thresh: float = Field(default=0.1, examples=[0.1])
    box_thresh: float = Field(default=0.1, examples=[0.1])

# =========================
# OUTPUT SCHEMAS
# =========================
class HealthModels(BaseModel):
    detection: Dict[str, bool]
    recognition: Dict[str, bool]

class HealthOut(BaseModel):
    status: str
    engine: str
    models_available: HealthModels
    device: str

class RecognitionOut(BaseModel):
    name: str = Field(..., examples=["example.jpg"])
    value: str = Field(..., examples=["Hello"])
    confidence: float = Field(..., examples=[0.99])


class DetectionOut(BaseModel):
    name: str = Field(..., examples=["example.jpg"])
    geometries: list[list[float]] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])


class OCRWord(BaseModel):
    value: str = Field(..., examples=["example"])
    geometry: list[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    confidence: float = Field(..., examples=[0.99])
    crop_orientation: dict[str, Any] = Field(..., examples=[{"value": 0, "confidence": None}])


class OCRLine(BaseModel):
    geometry: list[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    words: list[OCRWord] = Field(
        ...,
        examples=[
            {
                "value": "example",
                "geometry": [0.0, 0.0, 0.0, 0.0],
                "objectness_score": 0.99,
                "confidence": 0.99,
                "crop_orientation": {"value": 0, "confidence": None},
            }
        ],
    )


class OCRBlock(BaseModel):
    geometry: list[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    lines: list[OCRLine] = Field(
        ...,
        examples=[
            {
                "geometry": [0.0, 0.0, 0.0, 0.0],
                "objectness_score": 0.99,
                "words": [
                    {
                        "value": "example",
                        "geometry": [0.0, 0.0, 0.0, 0.0],
                        "confidence": 0.99,
                        "crop_orientation": {"value": 0, "confidence": None},
                    }
                ],
            }
        ],
    )


class OCRPage(BaseModel):
    blocks: list[OCRBlock] = Field(
        ...,
        examples=[
            {
                "geometry": [0.0, 0.0, 0.0, 0.0],
                "objectness_score": 0.99,
                "lines": [
                    {
                        "geometry": [0.0, 0.0, 0.0, 0.0],
                        "objectness_score": 0.99,
                        "words": [
                            {
                                "value": "example",
                                "geometry": [0.0, 0.0, 0.0, 0.0],
                                "objectness_score": 0.99,
                                "confidence": 0.99,
                                "crop_orientation": {"value": 0, "confidence": None},
                            }
                        ],
                    }
                ],
            }
        ],
    )


class OCROut(BaseModel):
    name: str = Field(..., examples=["example.jpg"])
    orientation: dict[str, float | None] = Field(..., examples=[{"value": 0.0, "confidence": 0.99}])
    language: dict[str, str | float | None] = Field(..., examples=[{"value": "en", "confidence": 0.99}])
    dimensions: tuple[int, int] = Field(..., examples=[(100, 100)])
    items: list[OCRPage] = Field(
        ...,
        examples=[
            {
                "geometry": [0.0, 0.0, 0.0, 0.0],
                "objectness_score": 0.99,
                "lines": [
                    {
                        "geometry": [0.0, 0.0, 0.0, 0.0],
                        "objectness_score": 0.99,
                        "words": [
                            {
                                "value": "example",
                                "geometry": [0.0, 0.0, 0.0, 0.0],
                                "objectness_score": 0.99,
                                "confidence": 0.99,
                                "crop_orientation": {"value": 0, "confidence": None},
                            }
                        ],
                    }
                ],
            }
        ],
    )


class ReadIn(BaseModel):
    document_class: str = Field(..., examples=["FACTURE"])
    gliner_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class EntityOut(BaseModel):
    label: str
    text: str
    start: int
    end: int
    score: float


class FieldResult(BaseModel):
    label: str
    raw: str
    value: str | None = None          # normalisé
    score: float
    status: str                       # OK | LOW_CONFIDENCE | INVALID
    reasons: list[str] = []
    extra: dict[str, Any] = {}


class ReadOut(BaseModel):
    name: str
    text: str
    entities: List[EntityOut]
    fields_validated: list[FieldResult] | None = None

