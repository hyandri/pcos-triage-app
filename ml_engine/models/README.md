# Pre-trained ML model files

Place the following serialized scikit-learn models in this directory:

- `symptom_pcos_model.pkl` — Stage 1 symptom triage model (10 features)
- `full_pcos_model.pkl` — Stage 2 full clinical model (33 features)

These files are loaded by `ml_engine/utils.py` in Phase 3. They are gitignored by default; copy your trained artifacts here for local development and Docker.
