from fastapi.testclient import TestClient
from app.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "SupportIQ API"
        assert "health" in data

        root_res = client.get("/")
        assert root_res.status_code == 200


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["models"]["trained_models_available"] is True


def test_model_metrics():
    with TestClient(app) as client:
        response = client.get("/api/model-metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert "category_model" in data["metrics"]
        assert "priority_model" in data["metrics"]


def test_submit_and_list_tickets():
    with TestClient(app) as client:
        payload = {
            "subject": "Delivery is delayed",
            "description": "My package tracking has not updated for three days and courier says delayed.",
        }
        create_res = client.post("/api/tickets", json=payload)
        assert create_res.status_code == 201
        ticket = create_res.json()
        assert ticket["category"] in ["delivery", "account", "billing", "technical", "refund"]
        assert ticket["priority"] in ["low", "medium", "high", "critical"]
        assert ticket["sentiment"] in ["positive", "neutral", "negative"]
        assert ticket["category_confidence"] > 0

        list_res = client.get("/api/tickets")
        assert list_res.status_code == 200
        tickets = list_res.json()
        assert len(tickets) >= 1
        assert any(t["id"] == ticket["id"] for t in tickets)


def test_analytics():
    with TestClient(app) as client:
        response = client.get("/api/analytics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_tickets"] >= 1
        assert len(data["category_counts"]) >= 1
        assert len(data["priority_counts"]) >= 1
        assert len(data["recent_tickets"]) >= 1


def test_invalid_ticket_rejection():
    with TestClient(app) as client:
        # Subject too short
        bad_subject = client.post("/api/tickets", json={"subject": "ab", "description": "Long enough description here"})
        assert bad_subject.status_code == 422

        # Description too short
        bad_desc = client.post("/api/tickets", json={"subject": "Valid subject", "description": "short"})
        assert bad_desc.status_code == 422
