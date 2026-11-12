import pytest
from pydantic import ValidationError

from app.schemas import TicketCreate


def test_ticket_create_accepts_valid_input():
    ticket = TicketCreate(
        subject="Payment charged twice",
        description="My card was charged two times for the same order.",
    )
    assert ticket.subject == "Payment charged twice"


def test_ticket_create_rejects_short_description():
    with pytest.raises(ValidationError):
        TicketCreate(subject="Payment issue", description="Short")
