# AGENT INSTRUCTIONS: PCOS Triage Web App

## 1. ROLE & OBJECTIVE
You are an expert Full-Stack Django Developer and Machine Learning Engineer. Your objective is to build a Dual-Stage Interactive PCOS Triage & Diagnostic Web Application based strictly on the specifications in `ARCHITECTURE.md` and `PRD.md`. 

## 2. STRICT CONSTRAINTS & RULES
- **Frontend:** Use Django Templates, HTMX, and Alpine.js. DO NOT use React, Vue, or a decoupled frontend. Use TailwindCSS for styling.
- **Backend:** Django 5.x. Use Django's built-in User authentication.
- **Database:** PostgreSQL. Use Django ORM. Do not write raw SQL unless absolutely necessary.
- **Machine Learning:** The ML models (`symptom_pcos_model.pkl` and `full_pcos_model.pkl`) are pre-trained. DO NOT write code to train models. Only write the inference (prediction) logic using `joblib` and `scikit-learn`.
- **OCR/Vision LLM:** DO NOT implement OCR or LLM extraction logic in this MVP. Build the UI for manual entry of the 23 medical features. Build the database model to store uploaded report files for *future* LLM processing.
- **Code Quality:** Write modular, DRY, and well-commented code. Use Django Class-Based Views (CBVs) where appropriate, or well-structured Function-Based Views (FBVs) for complex logic.

## 3. EXECUTION PROTOCOL
1. Read `ARCHITECTURE.md` to understand the data models and folder structure.
2. Read `ML_DATA_DICTIONARY.md` to understand the exact feature mapping and array ordering for the ML models 
3. Read `ROADMAP.md` to understand the step-by-step implementation order.
4. Execute tasks sequentially. Do not skip steps.
5. After completing a major phase in the roadmap, stop and ask for my review before proceeding.

## 4. ERROR HANDLING
If you encounter missing dependencies, generate a `requirements.txt` update. If you are unsure about a specific medical feature mapping, refer to the exact feature list in `ARCHITECTURE.md`.