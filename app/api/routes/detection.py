from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.schemas import DetectionIn, DetectionOut
from app.api.utils import get_documents, resolve_geometry
from app.api.vision import init_predictor

router = APIRouter()


@router.post(
        "/",
        response_model=list[DetectionOut],
        status_code=status.HTTP_200_OK,
        summary="Perform text detection",
)
async def text_detection(
    request: DetectionIn = Depends(),
    files: list[UploadFile] = [File(...)]
) -> list[DetectionOut]:
    """Runs docTR text detection model to analyze the input image"""
    try:
        predictor = init_predictor(request)
        content, filenames = await get_documents(files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return [
        DetectionOut(
            name=filename,
            geometries=[
                geom[:-1].tolist() if geom.shape[-1] == 5 else geom[:4].tolist()
                for geom in page
            ],
        )
        for page, filename in zip(
            predictor(content),
            filenames,
        )
    ]