# SYSTEM ARCHITECTURE

## 1. Tech Stack
- **Backend:** Python 3.11+, Django 5.x
- **Frontend:** Django Templates, HTMX (for dynamic partial updates), Alpine.js (for lightweight UI state like sliders/toggles), TailwindCSS.
- **Database:** PostgreSQL
- **Containerization:** Docker & Docker Compose (for local dev and deployment)
- **ML Inference:** `scikit-learn`, `joblib`, `numpy`, `pandas`

## 2. Directory Structure
```text
pcos_app/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── manage.py
├── core/                   # Main Django project settings
├── accounts/               # User auth, profiles, history
├── triage/                 # Stage 1: Symptom assessment logic & views
├── clinical/               # Stage 2: Medical assessment logic & views
├── ml_engine/              # Model loading, inference, SHAP values
│   ├── models/             # Directory for .pkl files
│   └── utils.py            # Prediction functions
├── templates/              # Global HTML templates
└── static/                 # CSS, JS, Images