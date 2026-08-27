# ML DATA DICTIONARY & FEATURE MAPPING

**Critical contract:** The pre-trained models require inputs in an exact feature order. The frontend may use readable state keys, but the backend/model adapter must map values to the exact original model column names before inference.

## 1. Model A: `symptom_pcos_model.pkl`

**Usage:** Stage 1 Quick Symptom Assessment. **Total features:** 10.

| Position | Exact model feature | Frontend/state key | Input |
|---:|---|---|---|
| 1 | `Age (yrs)` | `Age` | Numeric |
| 2 | `Weight (Kg)` | `Weight` | Numeric |
| 3 | `Height(Cm)` | `Height` | Numeric |
| 4 | `BMI` | `BMI` | Numeric, calculated or entered |
| 5 | `Cycle(R/I)` | `Cycle(R/I)` | Regular = 2, Irregular = 4 |
| 6 | `Cycle length(days)` | `Cycle length` | Numeric |
| 7 | `Marraige Status (Yrs)` | `Marraige Status` | Numeric |
| 8 | `Weight gain(Y/N)` | `Weight gain` | Yes = 1, No = 0 |
| 9 | `hair growth(Y/N)` | `hair growth` | Yes = 1, No = 0 |
| 10 | `Skin darkening (Y/N)` | `Skin darkening` | Yes = 1, No = 0 |

## 2. Model B: `full_pcos_model.pkl`

**Usage:** Stage 2 Full Clinical Assessment. **Total features:** 33.

The following is the complete, verified order extracted from the trained artifact:

| Position | Exact model feature | Frontend/state key | Input |
|---:|---|---|---|
| 1 | `Age (yrs)` | `Age` | Numeric |
| 2 | `Weight (Kg)` | `Weight` | Numeric |
| 3 | `Height(Cm)` | `Height` | Numeric |
| 4 | `BMI` | `BMI` | Numeric, calculated from weight and height |
| 5 | `Hb(g/dl)` | `Hb(g/dl)` | Numeric |
| 6 | `Cycle(R/I)` | `Cycle(R/I)` | Regular = 2, Irregular = 4 |
| 7 | `Cycle length(days)` | `Cycle length` | Numeric |
| 8 | `Marraige Status (Yrs)` | `Marraige Status` | Numeric |
| 9 | `No. of aborptions` | `No. of aborptions` | Numeric |
| 10 | `I   beta-HCG(mIU/mL)` | `I   beta-HCG(mIU/mL)` | Numeric |
| 11 | `II    beta-HCG(mIU/mL)` | `II    beta-HCG(mIU/mL)` | Numeric |
| 12 | `FSH(mIU/mL)` | `FSH(mIU/mL)` | Numeric |
| 13 | `LH(mIU/mL)` | `LH(mIU/mL)` | Numeric |
| 14 | `FSH/LH` | `FSH/LH` | Numeric, calculated or entered |
| 15 | `Hip(inch)` | `Hip(inch)` | Numeric |
| 16 | `Waist(inch)` | `Waist(inch)` | Numeric |
| 17 | `Waist:Hip Ratio` | `Waist:Hip Ratio` | Numeric, calculated or entered |
| 18 | `TSH (mIU/L)` | `TSH (mIU/L)` | Numeric |
| 19 | `AMH(ng/mL)` | `AMH(ng/mL)` | Numeric |
| 20 | `PRL(ng/mL)` | `PRL(ng/mL)` | Numeric |
| 21 | `Vit D3 (ng/mL)` | `Vit D3 (ng/mL)` | Numeric |
| 22 | `PRG(ng/mL)` | `PRG(ng/mL)` | Numeric |
| 23 | `RBS(mg/dl)` | `RBS(mg/dl)` | Numeric |
| 24 | `Weight gain(Y/N)` | `Weight gain` | Yes = 1, No = 0 |
| 25 | `hair growth(Y/N)` | `hair growth` | Yes = 1, No = 0 |
| 26 | `Skin darkening (Y/N)` | `Skin darkening` | Yes = 1, No = 0 |
| 27 | `Fast food (Y/N)` | `Fast food (Y/N)` | Yes = 1, No = 0 |
| 28 | `Follicle No. (L)` | `Follicle No. (L)` | Numeric |
| 29 | `Follicle No. (R)` | `Follicle No. (R)` | Numeric |
| 30 | `Avg. F size (L) (mm)` | `Avg. F size (L) (mm)` | Numeric |
| 31 | `Avg. F size (R) (mm)` | `Avg. F size (R) (mm)` | Numeric |
| 32 | `Endometrium (mm)` | `Endometrium (mm)` | Numeric |
| 33 | `Avg_Follicle_Count` | `Avg_Follicle_Count` | Numeric |

## 3. Frontend and backend payload contract

The clinical frontend must submit all 33 values using the clean state keys defined above:

```json
{
  "assessment_type": "clinical",
  "features": {
    "Age": 25,
    "Weight": 70.5,
    "Height": 165.0,
    "BMI": 25.9,
    "Hb(g/dl)": 13.0,
    "Cycle(R/I)": "Irregular",
    "Cycle length": 35,
    "Marraige Status": 2,
    "No. of aborptions": 0,
    "I   beta-HCG(mIU/mL)": 1.0,
    "II    beta-HCG(mIU/mL)": 1.0,
    "FSH(mIU/mL)": 6.0,
    "LH(mIU/mL)": 8.0,
    "FSH/LH": 0.75,
    "Hip(inch)": 40.0,
    "Waist(inch)": 34.0,
    "Waist:Hip Ratio": 0.85,
    "TSH (mIU/L)": 2.0,
    "AMH(ng/mL)": 4.0,
    "PRL(ng/mL)": 12.0,
    "Vit D3 (ng/mL)": 30.0,
    "PRG(ng/mL)": 1.0,
    "RBS(mg/dl)": 90.0,
    "Weight gain": false,
    "hair growth": false,
    "Skin darkening": false,
    "Fast food (Y/N)": false,
    "Follicle No. (L)": 8,
    "Follicle No. (R)": 8,
    "Avg. F size (L) (mm)": 6.0,
    "Avg. F size (R) (mm)": 6.0,
    "Endometrium (mm)": 7.0,
    "Avg_Follicle_Count": 8.0
  }
}
```

The application must map these clean frontend keys to the exact model column names above and preserve the listed order. Missing or unexpected fields must be rejected rather than silently reordered or guessed.
