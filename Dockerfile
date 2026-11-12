# ==============================================================================
# Stage 1: Build the React frontend
# ==============================================================================
FROM node:20-alpine AS frontend-builder
WORKDIR /build

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Production Python Backend + ML Models + Static Assets
# ==============================================================================
FROM python:3.11-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    KMP_DUPLICATE_LIB_OK=TRUE

# Install OpenMP runtime and dependencies for XGBoost and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy application source code
COPY backend/ ./backend/
COPY pytest.ini ./

# Copy compiled frontend from Stage 1 into the location expected by FastAPI
COPY --from=frontend-builder /build/dist ./frontend/dist

# Pre-train models and download sentiment model weights at build time so the container starts instantly
RUN python backend/scripts/generate_sample_data.py && \
    python backend/scripts/train.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --app-dir backend"]
