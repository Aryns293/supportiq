from fastapi import APIRouter, HTTPException, Query

from app.database import create_ticket, list_tickets
from app.schemas import TicketCreate, TicketResponse
from app.services.model_service import ModelNotReadyError, predict_ticket

router = APIRouter(prefix="/api/tickets", tags=["tickets"])


@router.post("", response_model=TicketResponse, status_code=201)
def submit_ticket(payload: TicketCreate):
    try:
        prediction = predict_ticket(payload.subject, payload.description)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "A prediction model could not run. Check network access for the first "
                f"sentiment-model download and inspect the backend logs. Details: {exc}"
            ),
        ) from exc

    row = create_ticket(payload.model_dump(), prediction)
    return TicketResponse(**row)


@router.get("", response_model=list[TicketResponse])
def get_tickets(limit: int = Query(default=20, ge=1, le=100)):
    return [TicketResponse(**row) for row in list_tickets(limit=limit)]
