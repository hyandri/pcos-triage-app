import joblib

# Load the clinical model
model = joblib.load('ml_engine/models/full_pcos_model.pkl')

# Try to extract feature names from the pipeline
if hasattr(model, 'feature_names_in_'):
    print("FOUND FEATURES:", model.feature_names_in_)
elif hasattr(model, 'named_steps'):
    for name, step in model.named_steps.items():
        if hasattr(step, 'feature_names_in_'):
            print(f"FOUND FEATURES in step '{name}':", step.feature_names_in_)
            break
else:
    print("Could not automatically find feature names. Check pipeline steps.")