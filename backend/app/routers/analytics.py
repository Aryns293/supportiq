import json

from fastapi import APIRouter

from app.config import METRICS_PATH
from app.database import get_analytics
from app.schemas import AnalyticsResponse

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
def analytics():
    return AnalyticsResponse(**get_analytics())


@router.get("/model-metrics")
def model_metrics():
    if not METRICS_PATH.exists():
        return {"available": False, "metrics": None}
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return {"available": True, "metrics": json.load(file)}
