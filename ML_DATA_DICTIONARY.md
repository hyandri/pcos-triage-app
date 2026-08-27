# ML DATA DICTIONARY & FEATURE MAPPING

**CRITICAL INSTRUCTION FOR AGENT:** The Machine Learning models require inputs in an EXACT, STRICT ORDER. If the array order is wrong, the predictions will be completely invalid. You must map the frontend form fields to the exact Python list order defined below.

## 1. Model A: `symptom_pcos_model.pkl` (10 Features)
**Usage:** Stage 1 Triage (Quick Symptom Assessment)
**Total Features:** 10

**Strict Array Order:**
1. `Age` (Numeric: Integer/Float)
2. `Weight` (Numeric: Float, in Kg)
3. `Height` (Numeric: Float, in Cm)
4. `BMI` (Numeric: Float, Auto-calculated or manual)
5. `Cycle(R/I)` (Categorical: Regular=2, Irregular=4)
6. `Cycle length` (Numeric: Integer, in days)
7. `Marraige Status` (Numeric: Integer, in years)
8. `Weight gain` (Boolean: Yes=1, No=0)
9. `hair growth` (Boolean: Yes=1, No=0)
10. `Skin darkening` (Boolean: Yes=1, No=0)

---

## 2. Model B: `full_pcos_model.pkl` (33 Features)
**Usage:** Stage 2 Clinical Assessment
**Total Features:** 33 (The 10 symptoms from Model A + 23 Medical Features)

**Strict Array Order:**
*Features 1-10 are exactly the same as Model A.*

**Features 11-33 (Medical Diagnostics):**
11. `Follicle No. (R)` (Numeric: Integer)
12. `Follicle No. (L)` (Numeric: Integer)
13. `Avg_Follicle_Count` (Numeric: Float)
14. `Avg. F size (L) (mm)` (Numeric: Float)
15. `Avg. F size (R) (mm)` (Numeric: Float)
16. `Endometrium (mm)` (Numeric: Float)
17. `AMH(ng/mL)` (Numeric: Float)
18. `FSH(mIU/mL)` (Numeric: Float)
19. `LH(mIU/mL)` (Numeric: Float)
20. `FSH/LH` (Numeric: Float, Auto-calculated or manual)
21. `TSH (mIU/L)` (Numeric: Float)
22. `PRL(ng/mL)` (Numeric: Float)
23. `Vit D3 (ng/mL)` (Numeric: Float)
24. `PRG(ng/mL)` (Numeric: Float)
25. `RBS(mg/dl)` (Numeric: Float)
26. `Fast food (Y/N)` (Boolean: Yes=1, No=0)
*(Note to Agent: The blueprint specifies 23 medical features. Ensure you verify the exact remaining 7 features from the training dataset column names to complete the 33-feature array. Do not guess the names.)*

---

## 3. Implementation Rules for the Agent
1. **Frontend Forms:** Build the HTML forms (using HTMX/Alpine) to match these exact variable names and input types.
2. **Backend Payload:** When the form is submitted, the Django view must extract these values and construct a Python list in the **EXACT ORDER** shown above.
3. **Numpy Conversion:** Convert the list to a 2D Numpy array (e.g., `np.array([features]).reshape(1, -1)`) before passing it to `model.predict_proba()`.