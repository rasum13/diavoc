import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from data_loader import load_and_process_data, split_non_iid

# --- 1. Federated Training & Aggregation ---
def train_federated_system(X, y):
    """
    Simulates clients with local tuning and returns a saved global ensemble.
    """
    # Outer 10-Fold CV to verify the 90% accuracy goal
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_fold_score = 0
    final_model = None

    print(f"Starting Federated Tournament Training...")
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Split among 5 clients
        client_shards = split_non_iid(X_train, y_train, num_clients=5)
        local_experts = []

        for j, (X_c, y_c) in enumerate(client_shards):
            # Each client runs GridSearchCV to find their best local parameters
            # Comparing SVC and RF for the best voice-to-diabetes mapping
            param_grid = [
                {'estimator': [SVC(probability=True, class_weight='balanced')], 
                 'params': {'C': [1, 10], 'kernel': ['rbf']}},
                {'estimator': [RandomForestClassifier(class_weight='balanced')], 
                 'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}}
            ]
            
            best_local = None
            max_f1 = -1
            for config in param_grid:
                gs = GridSearchCV(config['estimator'][0], config['params'], cv=3, scoring='f1_weighted')
                gs.fit(X_c, y_c)
                if gs.best_score_ > max_f1:
                    max_f1 = gs.best_score_
                    best_local = gs.best_estimator_
            
            local_experts.append((f'client_{j}', best_local))

        # Server: Create a Global Voting Ensemble from local experts
        global_ensemble = VotingClassifier(estimators=local_experts, voting='soft')
        global_ensemble.fit(X_train, y_train)

        # Evaluate this fold
        preds = global_ensemble.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Fold {i+1:02d} Accuracy: {acc:.4f}")
        
        if acc > best_fold_score:
            best_fold_score = acc
            final_model = global_ensemble

    return final_model

# --- 2. Save Model for Deployment ---
def save_system(model, filename="diavoc_final_model.joblib"):
    # Note: In a real scenario, you'd also save the Scaler and PCA from data_loader
    joblib.dump(model, filename)
    print(f"\n✅ Production model saved as: {filename}")

# --- 3. Inference Function ---
def infer(model_path, patient_features):
    """
    patient_features: A single processed numpy array of 333 features.
    """
    model = joblib.load(model_path)
    # Ensure input is 2D
    features = patient_features.reshape(1, -1)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    result = "DIABETIC" if prediction == 1 else "HEALTHY"
    return result, probability

# --- Execution ---
if __name__ == "__main__":
    # Load your balanced dataset (607 samples, 333 features)
    X, y = load_and_process_data('male_embeddings.pkl', 'female_embeddings.pkl')
    
    # Train
    fed_model = train_federated_system(X, y)
    
    # Save
    save_system(fed_model)
    
    # Test Inference on one sample
    sample_idx = 0
    test_patient = X[sample_idx]
    label, confidence = infer("diavoc_final_model.joblib", test_patient)
    
    print(f"\n--- Inference Test ---")
    print(f"Sample Result: {label}")
    print(f"Model Confidence: {confidence:.2%}")
    print(f"Actual Label: {'DIABETIC' if y[sample_idx] == 1 else 'HEALTHY'}")