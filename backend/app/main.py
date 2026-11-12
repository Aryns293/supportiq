from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, PROJECT_DIR
from app.database import init_db
from app.routers import analytics, tickets
from app.services.model_service import model_status


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SupportIQ API",
    description="Support ticket classification, priority prediction and sentiment analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(tickets.router)
app.include_router(analytics.router)


@app.get("/api")
def api_info():
    return {"name": "SupportIQ API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    return {"status": "ok", "models": model_status()}


FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"name": "SupportIQ API", "docs": "/docs", "health": "/api/health"}

