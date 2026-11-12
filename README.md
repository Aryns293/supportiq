# SupportIQ 🧠🎫

> **AI-Assisted Customer Support Ticket Intelligence & Triage Platform**  
> Classify ticket categories, estimate urgency, and analyze customer sentiment in real time using scikit-learn, XGBoost, DistilBERT, FastAPI, and React.

[![CI](https://github.com/Aryns293/supportiq/actions/workflows/ci.yml/badge.svg)](https://github.com/Aryns293/supportiq/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/Aryns293/supportiq/actions/workflows/deploy-pages.yml/badge.svg)](https://aryns293.github.io/supportiq/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://github.com/Aryns293/supportiq/blob/main/Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Aryns293/supportiq)

---

## 🌐 Live Demo & Endpoints

| Resource | URL | Description |
|---|---|---|
| **Live Web Application (Render)** | [https://supportiq-rndw.onrender.com](https://supportiq-rndw.onrender.com) | Fullstack dashboard & live inference |
| **Interactive API Documentation** | [https://supportiq-rndw.onrender.com/docs](https://supportiq-rndw.onrender.com/docs) | Swagger UI for testing API routes |
| **API Health Check** | [https://supportiq-rndw.onrender.com/api/health](https://supportiq-rndw.onrender.com/api/health) | Real-time model & service status |
| **Frontend Mirror (GitHub Pages)** | [https://aryns293.github.io/supportiq/](https://aryns293.github.io/supportiq/) | Static client mirror |
| **GitHub Repository** | [https://github.com/Aryns293/supportiq](https://github.com/Aryns293/supportiq) | Source code, models, and workflows |

---

## 📌 Introduction

**SupportIQ** is an end-to-end Machine Learning and Fullstack application designed to streamline customer support operations. When a customer submits an inquiry, SupportIQ analyzes the text simultaneously across three distinct ML dimensions:

1. **Category Classification:** Identifies domain context (`billing`, `technical`, `account`, `delivery`, `refund`) using sublinear TF-IDF word & n-gram vectors with balanced multiclass **Logistic Regression**.
2. **Priority Prediction:** Estimates urgency level (`low`, `medium`, `high`, `critical`) using gradient-boosted decision trees via **XGBoost** with sample weight balancing.
3. **Sentiment Analysis:** Gauges customer emotion (`positive`, `neutral`, `negative`) using a fine-tuned multilingual **DistilBERT** transformer pipeline.

Predictions, confidence scores, and historical metrics are saved to SQLite and surfaced through an analytical dashboard built in **React, TypeScript, and Recharts**.

---

## 🚀 Key Features

* **Real-Time Multi-Model Inference:** Concurrent execution of TF-IDF feature extractors, linear models, tree-based models, and deep learning transformers on incoming text.
* **Interactive Analytics Dashboard:** Real-time urgency mix pie charts, category distribution bar charts, held-out validation metric cards, and recent ticket feeds.
* **Unified Single-Port Production Serving:** FastAPI statically mounts and serves the compiled React Single Page Application while managing REST endpoints (`/api/...`).
* **Robust Mac & Linux OpenMP Portability:** Configured with runtime isolation guards preventing dual OpenMP runtime collisions between PyTorch and XGBoost.
* **Automated CI/CD:** Continuous Integration on GitHub Actions running automated dataset generation, model retraining, pytest regression tests, and TypeScript compilation.
* **1-Click Containerized Cloud Deployment:** Ready for instant zero-config deployment on Render, Railway, Fly.io, or Docker.

---

## 🏗️ Architecture & Model Pipeline

```text
[Customer Ticket]
  │ (Subject + Description)
  ├──► TF-IDF Vectorizer ────► Logistic Regression ──► Predicted Category + Confidence
  │
  ├──► TF-IDF Vectorizer ────► XGBoost Classifier  ──► Predicted Priority + Confidence
  │
  └──► DistilBERT Transformer Pipeline (HF)        ──► Customer Sentiment + Confidence
                                                               │
                                                               ▼
                                                     [SQLite Database]
                                                               │
                                                               ▼
                                                 [React Analytics Dashboard]
```

### Model Specifications

| Model | Technique | Output Classes | Purpose |
|---|---|---|---|
| **Category** | Sublinear TF-IDF (1-2 n-grams) + Logistic Regression | `billing`, `technical`, `account`, `delivery`, `refund` | Route tickets to the right department |
| **Priority** | Sublinear TF-IDF + XGBoost Multi:Softprob | `low`, `medium`, `high`, `critical` | Triage urgent issues before SLA breach |
| **Sentiment** | Pretrained Multilingual DistilBERT (`lxyuan/...`) | `positive`, `neutral`, `negative` | Detect frustrated or churning users |

---

## 📁 Project Directory Structure

```text
supportiq/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Automated testing and training CI pipeline
│       └── deploy-pages.yml    # Continuous GitHub Pages deployment
├── Dockerfile                  # Multi-stage production container
├── render.yaml                 # 1-click Render blueprint specification
├── pytest.ini                  # Pytest configuration
├── README.md
│
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── main.py             # FastAPI app with static SPA mount
│   │   ├── config.py           # Environment variables and path resolution
│   │   ├── database.py         # SQLite connection and analytics aggregation
│   │   ├── schemas.py          # Pydantic request/response validation
│   │   ├── routers/
│   │   │   ├── tickets.py      # Ticket creation and retrieval APIs
│   │   │   └── analytics.py    # Analytics metrics & model evaluation APIs
│   │   └── services/
│   │       └── model_service.py # Model loading, caching, and inference
│   ├── data/
│   │   └── sample_tickets.csv  # 700-row synthetic demo training set
│   ├── models/                 # Generated serialized .joblib model bundle
│   ├── reports/                # Generated evaluation report JSON
│   ├── scripts/
│   │   ├── generate_sample_data.py # Demo dataset synthesizer
│   │   └── train.py            # Model training & evaluation pipeline
│   └── tests/
│       ├── test_api.py         # End-to-end API integration tests
│       └── test_schemas.py     # Schema validation tests
│
└── frontend/
    ├── package.json
    ├── index.html
    ├── vite.config.ts
    ├── tsconfig.json
    └── src/
        ├── App.tsx             # Interactive dashboard UI & visualizations
        ├── main.tsx
        ├── styles.css          # Design system styling
        └── lib/
            └── api.ts          # Type-safe API client
```

---

## 🛠️ Local Development Setup

### 1. Prerequisites
* **Python:** 3.10 or 3.11
* **Node.js:** 20 or newer
* **macOS only:** Run `brew install libomp` to install the OpenMP C runtime for XGBoost.

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/Aryns293/supportiq.git
cd supportiq

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt

# Generate dataset and train both ML models
python backend/scripts/generate_sample_data.py
python backend/scripts/train.py

# Run tests
pytest
```

### 3. Start Development Servers
**Terminal 1 — FastAPI Backend:**
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir backend --port 8000
```

**Terminal 2 — React Frontend:**
```bash
cd frontend
npm install
npm run dev
```

* **Frontend:** [http://localhost:5173](http://localhost:5173)
* **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Production Deployment

### 1. Cloud Deployment (Render)
SupportIQ is pre-configured with [render.yaml](render.yaml). Click the deploy button:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Aryns293/supportiq)

### 2. Docker Container
The included multi-stage [Dockerfile](Dockerfile) compiles the React frontend, installs Python 3.11 with OpenMP runtime, pre-trains models, and serves both frontend and API from a single container:

```bash
docker build -t supportiq .
docker run -p 8000:8000 supportiq
```

Open [http://localhost:8000](http://localhost:8000).

---

## 👨‍💻 Author

**Aryan Sharma**

B.Tech, Delhi Technological University

- **GitHub:** [https://github.com/Aryns293](https://github.com/Aryns293)
- **LinkedIn:** [https://www.linkedin.com/in/aryan-sharma-29m/](https://www.linkedin.com/in/aryan-sharma-29m/)
- **Repository:** [https://github.com/Aryns293/supportiq](https://github.com/Aryns293/supportiq)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
