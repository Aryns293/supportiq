from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class TicketCreate(BaseModel):
    subject: str = Field(min_length=3, max_length=180)
    description: str = Field(min_length=8, max_length=4000)


class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subject: str
    description: str
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    sentiment: str
    sentiment_confidence: float
    created_at: datetime


class AnalyticsResponse(BaseModel):
    total_tickets: int
    category_counts: list[dict[str, str | int]]
    priority_counts: list[dict[str, str | int]]
    sentiment_counts: list[dict[str, str | int]]
    recent_tickets: list[TicketResponse]
