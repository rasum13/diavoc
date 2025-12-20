import joblib
import numpy as np
import pandas as pd

def load_inference_system(model_path="diavoc_final_model.joblib"):
    """Loads the saved federated model."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print("Error: Model file not found. Please run training first.")
        return None

def run_inference(patient_data, model_path="diavoc_final_model.joblib"):
    """
    patient_data: A dict containing:
        - 'age': int
        - 'bmi': float
        - 'gender': 'male' or 'female'
        - 'embeddings': list/array of 4096 BYOL features
    """
    # 1. Load the Model
    model = load_inference_system(model_path)
    if model is None: return

    # 2. Extract Tabular Features (Must match training logic exactly)
    gender_bin = 1 if patient_data['gender'].lower() == 'male' else 0
    risk_index = (patient_data['age'] * patient_data['bmi']) / 100.0
    
    # 3. Process Voice Embeddings
    # In a real app, you would also apply the same PCA and Scaler here
    # For this snippet, we assume patient_data['processed_features'] is already
    # scaled and PCA-transformed to match the 333 features.
    X_input = patient_data['processed_features'].reshape(1, -1)

    # 4. Get Probability and Prediction
    # Soft voting takes the average of all client models
    probability = model.predict_proba(X_input)[0][1]
    prediction = model.predict(X_input)[0]

    # 5. Clinical Interpretation
    status = "DIABETIC" if prediction == 1 else "HEALTHY"
    
    # Define Risk Tiers
    if probability > 0.85:
        risk_level = "High Risk (Immediate Follow-up Recommended)"
    elif probability > 0.50:
        risk_level = "Elevated Risk (Further Screening Advised)"
    else:
        risk_level = "Low Risk"

    return {
        "Diagnosis": status,
        "Confidence": f"{probability:.2%}",
        "Risk Tier": risk_level
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Mock data for a new patient
    # Note: 'processed_features' should be the result of your Scaler/PCA
    mock_patient = {
        'age': 45,
        'bmi': 28.5,
        'gender': 'female',
        'processed_features': np.random.randn(333) # Replace with real processed data
    }

    result = run_inference(mock_patient)
    
    print("\n--- DiaVoc Diagnostic Report ---")
    for key, value in result.items():
        print(f"{key}: {value}")
