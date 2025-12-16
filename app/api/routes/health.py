from fastapi import APIRouter, status

from app.api.schemas import HealthOut
from app.api.state import app_state

router = APIRouter()


@router.get(
    "/",
    response_model=HealthOut,
    status_code=status.HTTP_200_OK,
    summary="Return API and model status",
)
async def health_check() :
    return HealthOut(
        status="ok",
        engine=app_state["engineType"],
        models_available=app_state["models_available"],
        device=app_state["device"],
    )