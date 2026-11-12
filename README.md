# SupportIQ — ML-powered support ticket triage

SupportIQ is a deliberately medium-sized portfolio project. It accepts a support ticket and produces three predictions:

1. **Category** — trained by you with TF-IDF + Logistic Regression.
2. **Priority** — trained by you with TF-IDF + XGBoost.
3. **Sentiment** — produced by a pretrained three-class Hugging Face model (positive / neutral / negative).

It also stores analyzed tickets in SQLite and shows ticket distributions plus the evaluation report from the two trained models.

## What this project demonstrates

- Labeled tabular/text data preparation with Pandas.
- Text feature extraction with TF-IDF.
- Supervised classification with Logistic Regression and XGBoost.
- Stratified train/test splitting and model evaluation.
- Accuracy, macro F1, weighted F1, classification reports and confusion matrices.
- Loading serialized model artifacts and serving predictions through FastAPI.
- Calling a pretrained transformer for sentiment inference.
- A React + TypeScript dashboard that consumes the API.

## Important data and evaluation warning

The included `backend/data/sample_tickets.csv` is **synthetic demo data generated from templates**. It exists so that the end-to-end training and application flow can run immediately. Scores calculated on it are useful only to check that the pipeline works. They are not evidence of real-world accuracy and must not be presented as portfolio performance claims.

For a serious portfolio evaluation, replace it with a properly licensed, labeled support-ticket dataset. Your CSV must contain these columns:

| Column | Meaning | Example |
|---|---|---|
| `subject` | Ticket title | `Payment charged twice` |
| `description` | Customer's issue text | `My card was charged two times...` |
| `category` | Human-assigned category label | `billing` |
| `priority` | Human-assigned priority label | `high` |

You need enough examples in every category and priority class. Keep labels consistent and inspect class balance before training. Do not use the generated dataset as the final evaluation dataset.

## Project structure

```text
supportiq/
├── README.md
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── schemas.py
│   │   ├── routers/
│   │   │   ├── tickets.py
│   │   │   └── analytics.py
│   │   └── services/
│   │       └── model_service.py
│   ├── data/
│   │   └── sample_tickets.csv
│   ├── models/                 # trained .joblib model bundle is generated here
│   ├── reports/                # model_metrics.json is generated here
│   ├── scripts/
│   │   ├── generate_sample_data.py
│   │   └── train.py
│   └── tests/
│       └── test_schemas.py
└── frontend/
    ├── package.json
    ├── index.html
    ├── vite.config.ts
    ├── tsconfig.json
    └── src/
        ├── App.tsx
        ├── main.tsx
        ├── styles.css
        └── lib/api.ts
```

## Requirements

- Python 3.10 or newer; Python 3.11 is a good choice.
- Node.js 20 or newer and npm.
- Internet access for the first download of the pretrained sentiment model.
- The first sentiment inference downloads a model from Hugging Face. Its model card identifies the three labels as positive, neutral and negative and documents a Transformers pipeline example: <https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student>.

## Run locally on macOS / Linux

Run these commands from the `supportiq/` project root.

### 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

PyTorch / Transformers take more disk space than the other Python packages. On Apple Silicon, install in a native ARM Python environment rather than an x86/Rosetta environment.

### 2. Generate the demo dataset

```bash
python backend/scripts/generate_sample_data.py
```

The command creates 700 synthetic, labeled examples by default.

### 3. Train the two ML models

```bash
python backend/scripts/train.py
```

This creates:

- `backend/models/supportiq_models.joblib` — TF-IDF vectorizer, Logistic Regression category classifier, XGBoost priority classifier and label encoder.
- `backend/reports/model_metrics.json` — held-out test metrics and confusion matrices.

To train on your own correctly labeled dataset:

```bash
python backend/scripts/train.py --data backend/data/your_labeled_tickets.csv
```

The script uses a stratified train/test split. Feature extraction is fitted only on the training portion to avoid fitting the vocabulary on held-out examples.

### 4. Start the backend

Open Terminal 1 from the project root. Activate the environment if needed, then run:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir backend
```

Backend URLs:

- API: <http://localhost:8000>
- Swagger docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/health>

### 5. Start the frontend

Open Terminal 2 from the project root:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

The Vite development server calls `http://localhost:8000` by default. To change it, copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_URL`.

## Run locally on Windows PowerShell

From the `supportiq/` project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python backend/scripts/generate_sample_data.py
python backend/scripts/train.py
```

Start the backend in the first PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --app-dir backend
```

Start the frontend in a second PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | API and model-artifact status |
| `POST` | `/api/tickets` | Predict category, priority and sentiment, then store ticket |
| `GET` | `/api/tickets?limit=20` | List recent tickets |
| `GET` | `/api/analytics` | Ticket counts by category, priority and sentiment |
| `GET` | `/api/model-metrics` | Return the saved training evaluation report |

Example request:

```bash
curl -X POST http://localhost:8000/api/tickets \
  -H 'Content-Type: application/json' \
  -d '{
    "subject": "Payment charged twice",
    "description": "My card was charged two times for the same order. This is blocking an important task and needs prompt attention."
  }'
```

The API returns model predictions and confidence scores. These scores are not guaranteed to be calibrated probabilities. A human support agent should review predictions before taking action.

## Run tests

From the project root, with the virtual environment activated:

```bash
cd backend
python -m pytest
cd ..
```

## How the three prediction paths differ

### Category classifier — trained by this project

`subject + description` → TF-IDF sparse vectors → Logistic Regression → predicted category.

### Priority classifier — trained by this project

`subject + description` → same TF-IDF feature representation → XGBoost multi-class classifier → predicted priority.

### Sentiment classifier — pretrained, not trained by this project

`subject + description` → pretrained multilingual DistilBERT sentiment model → positive / neutral / negative label.

Do not describe the sentiment transformer as a model you trained. The category and priority models are the two models trained in this repository.

## Current limitations (intentional for a medium-sized project)

- No authentication or role-based access control.
- SQLite is used for local development; PostgreSQL can replace it later.
- No Redis, Celery, Kafka, microservices or Kubernetes.
- Predictions are synchronous; the pretrained sentiment model loads on the first prediction.
- The included training data is synthetic and must be replaced for meaningful evaluation.
- No automated retraining, drift monitoring or model registry.
- This is a portfolio/demo application, not a production support platform.

## Before putting it on your resume

1. Replace synthetic data with a licensed, representative labeled dataset.
2. Inspect label quality, duplicates, class imbalance and leakage.
3. Compare Logistic Regression against at least one meaningful baseline for category prediction.
4. Compare XGBoost against a simple baseline for priority prediction.
5. Report macro F1, per-class precision/recall, confusion matrices and the exact test protocol.
6. Keep the final test set untouched until model selection is complete.
7. Include the dataset source, license and limitations in your README.
8. Never copy demo metrics into your resume as real performance claims.
