# PROJECT ROADMAP

## Phase 1: Project Setup & Infrastructure
- [ ] Initialize Django project and apps (`accounts`, `triage`, `clinical`, `ml_engine`).
- [ ] Configure `settings.py` for PostgreSQL, TailwindCSS, and static/media files.
- [ ] Create `Dockerfile` and `docker-compose.yml` (Django + Postgres).
- [ ] Add placeholder `.pkl` files in `ml_engine/models/` (or provide instructions on where I will put the real ones).

## Phase 2: Database & Authentication
- [ ] Implement `UserProfile`, `AssessmentSession`, and `MedicalReportFile` models.
- [ ] Run migrations.
- [ ] Set up Django built-in auth (Login, Register, Logout views and templates).
- [ ] Create the main Dashboard view to list past `AssessmentSession` history.

## Phase 3: ML Engine Integration
- [ ] Create `ml_engine/utils.py` with functions to load models and run inference.
- [ ] Implement strict input validation to ensure the 10-feature and 33-feature arrays match the exact column order of the trained models.
- [ ] Write unit tests for the inference engine to ensure it returns correct shapes and probabilities.

## Phase 4: Stage 1 UI (Symptom Flashcards)
- [ ] Create base HTML templates with TailwindCSS, HTMX, and Alpine.js.
- [ ] Build the interactive flashcard UI (Binary toggles, Sliders, Numeric inputs).
- [ ] Implement HTMX endpoints to handle step-by-step progression without page reloads.
- [ ] Connect Stage 1 UI to `symptom_pcos_model.pkl` and save results to `AssessmentSession`.

## Phase 5: Stage 2 UI (Clinical Assessment)
- [ ] Build the multi-section form for the 23 medical features.
- [ ] Implement the file upload widget for `MedicalReportFile`.
- [ ] Connect Stage 2 UI to `full_pcos_model.pkl`.
- [ ] Ensure the UI gracefully handles missing medical data (e.g., if a user doesn't have an AMH test, how do we handle nulls? *Agent: implement median imputation or require specific fields*).

## Phase 6: Polish & Deployment Prep
- [ ] Create results display page (Risk Tier, Probability Score).
- [ ] Ensure all past assessments on the Dashboard link to their specific results pages.
- [ ] Write the `README.md` with local Docker setup instructions and DigitalOcean deployment steps.