"""
DiaVoc Federated Learning Training Pipeline
Implements FedAvg with MLP Classifier (Spec Section 9-10)
Uses true parameter averaging, not ensemble voting
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, classification_report, confusion_matrix)
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# SECTION 1: NON-IID DATA PARTITIONING (Spec Section 3)
# ============================================================================

def split_non_iid_dirichlet(X, y, num_clients=5, alpha=0.5, min_samples_per_class=2, random_state=42):
    """
    Split data among clients using Dirichlet distribution for non-IID partitioning.
    Ensures each client has minimum viable samples per class for stratified splitting.
    
    Args:
        X: Feature matrix (N, D)
        y: Labels (N,)
        num_clients: Number of federated clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
               α=0.5 (spec default) creates highly skewed distributions
        min_samples_per_class: Minimum samples per class each client should have
    
    Returns:
        list of tuples: [(X_client1, y_client1), ..., (X_clientN, y_clientN)]
    """
    np.random.seed(random_state)
    n_classes = len(np.unique(y))
    n_samples = len(y)
    
    # Initialize client data holders
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples among clients using Dirichlet
    for k in range(n_classes):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)
        
        # Reserve minimum samples for each client first
        reserved_per_client = min_samples_per_class
        total_reserved = reserved_per_client * num_clients
        
        if len(idx_k) < total_reserved:
            # Not enough samples - distribute evenly
            print(f"⚠️  Class {k}: Only {len(idx_k)} samples, distributing evenly")
            split_size = len(idx_k) // num_clients
            for i in range(num_clients):
                start = i * split_size
                end = start + split_size if i < num_clients - 1 else len(idx_k)
                client_indices[i].extend(idx_k[start:end])
        else:
            # Enough samples - first reserve minimum, then distribute rest with Dirichlet
            # Reserve minimum for each client
            for i in range(num_clients):
                client_indices[i].extend(idx_k[i * reserved_per_client:(i + 1) * reserved_per_client])
            
            # Remaining samples to distribute
            remaining_idx = idx_k[total_reserved:]
            
            if len(remaining_idx) > 0:
                # Sample from Dirichlet for remaining samples
                proportions = np.random.dirichlet(alpha * np.ones(num_clients))
                counts = (proportions * len(remaining_idx)).astype(int)
                counts[-1] = len(remaining_idx) - counts[:-1].sum()  # Adjust last
                
                # Distribute remaining
                start_idx = 0
                for i, count in enumerate(counts):
                    if count > 0:
                        client_indices[i].extend(remaining_idx[start_idx:start_idx + count])
                        start_idx += count
    
    # Create client datasets
    clients = []
    for i, indices in enumerate(client_indices):
        indices = np.array(indices)
        if len(indices) == 0:
            print(f"⚠️  Client {i} has no data, skipping...")
            continue
            
        clients.append((X[indices], y[indices]))
        
        # Print distribution stats
        class_dist = np.bincount(y[indices], minlength=n_classes)
        diabetic_pct = (class_dist[1]/len(indices)*100) if len(indices) > 0 else 0
        print(f"Client {i}: {len(indices):3d} samples | "
              f"Healthy: {class_dist[0]:3d}, Diabetic: {class_dist[1]:3d} | "
              f"Diabetic %: {diabetic_pct:5.1f}%")
    
    return clients


# ============================================================================
# SECTION 2: MODEL INITIALIZATION
# ============================================================================

def create_model(model_type='mlp', input_dim=None, random_state=42):
    """
    Create FL-compatible model (MLP, LogisticRegression, or LinearSVM).
    
    Args:
        model_type: 'mlp', 'logistic', or 'linear_svm'
        input_dim: Number of input features (required for MLP)
    
    Returns:
        sklearn classifier with extractable parameters
    """
    if model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Deeper network
            activation='relu',
            solver='adam',
            alpha=0.001,  # Increased regularization
            batch_size='auto',
            learning_rate='adaptive',  # Adaptive learning rate
            learning_rate_init=0.001,
            max_iter=100,  # More iterations
            random_state=random_state,
            warm_start=False,
            early_stopping=False,
            verbose=False,
            n_iter_no_change=20
        )
        
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,  # More iterations
            solver='lbfgs',
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
    elif model_type == 'linear_svm':
        model = LinearSVC(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            dual='auto',
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


# ============================================================================
# SECTION 3: FEDERATED AVERAGING (FedAvg) - Spec Section 10
# ============================================================================

def get_model_parameters(model):
    """Extract parameters from sklearn model."""
    if isinstance(model, MLPClassifier):
        # MLP: weights (coefs_) and biases (intercepts_)
        params = {
            'coefs': [w.copy() for w in model.coefs_],
            'intercepts': [b.copy() for b in model.intercepts_]
        }
    elif isinstance(model, (LogisticRegression, LinearSVC)):
        # Linear models: coef_ and intercept_
        params = {
            'coef': model.coef_.copy(),
            'intercept': model.intercept_.copy()
        }
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    return params


def set_model_parameters(model, params):
    """Set parameters to sklearn model."""
    if isinstance(model, MLPClassifier):
        model.coefs_ = [w.copy() for w in params['coefs']]
        model.intercepts_ = [b.copy() for b in params['intercepts']]
    elif isinstance(model, (LogisticRegression, LinearSVC)):
        model.coef_ = params['coef'].copy()
        model.intercept_ = params['intercept'].copy()
    
    return model


def federated_averaging(client_models, client_weights):
    """
    FedAvg: Weighted average of client model parameters.
    
    Formula (Spec Section 10):
        global_weights = Σ(client_weights_i × client_params_i) / Σ(client_weights_i)
    
    Args:
        client_models: List of trained client models
        client_weights: List of dataset sizes for each client
    
    Returns:
        dict: Averaged parameters
    """
    total_weight = sum(client_weights)
    
    # Get first model's structure
    first_params = get_model_parameters(client_models[0])
    
    # Initialize averaged parameters
    if 'coefs' in first_params:  # MLP
        avg_params = {
            'coefs': [np.zeros_like(w) for w in first_params['coefs']],
            'intercepts': [np.zeros_like(b) for b in first_params['intercepts']]
        }
        
        # Weighted sum
        for model, weight in zip(client_models, client_weights):
            params = get_model_parameters(model)
            for i in range(len(params['coefs'])):
                avg_params['coefs'][i] += params['coefs'][i] * weight / total_weight
                avg_params['intercepts'][i] += params['intercepts'][i] * weight / total_weight
    
    else:  # Linear models
        avg_params = {
            'coef': np.zeros_like(first_params['coef']),
            'intercept': np.zeros_like(first_params['intercept'])
        }
        
        # Weighted sum
        for model, weight in zip(client_models, client_weights):
            params = get_model_parameters(model)
            avg_params['coef'] += params['coef'] * weight / total_weight
            avg_params['intercept'] += params['intercept'] * weight / total_weight
    
    return avg_params


# ============================================================================
# SECTION 4: LOCAL TRAINING (Client-Side)
# ============================================================================

def train_local_model(model, X_train, y_train, X_val, y_val, init_params=None):
    """
    Train model on local client data.
    
    Args:
        model: sklearn classifier
        X_train, y_train: Training data
        X_val, y_val: Validation data
        init_params: Optional initial parameters from global model
    
    Returns:
        tuple: (trained_model, metrics_dict)
    """
    # For MLP with initial parameters, properly initialize
    if init_params is not None and isinstance(model, MLPClassifier):
        # Find samples with both classes for proper initialization
        unique_classes = np.unique(y_train)
        if len(unique_classes) == 2:
            init_indices = []
            for cls in unique_classes:
                cls_indices = np.where(y_train == cls)[0]
                init_indices.extend(cls_indices[:min(10, len(cls_indices))])
            init_indices = np.array(init_indices)
        else:
            init_indices = np.arange(min(20, len(X_train)))
        
        # Initialize structure
        model.fit(X_train[init_indices], y_train[init_indices])
        
        # Transfer global parameters
        model.coefs_ = [w.copy() for w in init_params['coefs']]
        model.intercepts_ = [b.copy() for b in init_params['intercepts']]
        model.warm_start = True
        model.max_iter = 50  # Continue training
    
    # Train on full data
    model.fit(X_train, y_train)
    
    # Reset settings
    if isinstance(model, MLPClassifier):
        model.warm_start = False
        model.max_iter = 100
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    metrics = {
        'train_acc': accuracy_score(y_train, y_pred_train),
        'train_f1': f1_score(y_train, y_pred_train, average='weighted', zero_division=0),
        'val_acc': accuracy_score(y_val, y_pred_val),
        'val_f1': f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
    }
    
    return model, metrics


# ============================================================================
# SECTION 5: FEDERATED TRAINING LOOP
# ============================================================================

def train_federated(clients_data, model_type='mlp', num_rounds=10, 
                    val_split=0.2, random_state=42):
    """
    Main federated learning training loop with FedAvg.
    
    Args:
        clients_data: List of (X_client, y_client) tuples
        model_type: 'mlp', 'logistic', or 'linear_svm'
        num_rounds: Number of federated rounds (Spec: 10)
        val_split: Validation split ratio per client
        random_state: Random seed
    
    Returns:
        dict: {
            'global_model': Final global model,
            'history': Training metrics per round,
            'client_models': Final local models
        }
    """
    num_clients = len(clients_data)
    input_dim = clients_data[0][0].shape[1]
    
    # Initialize global model
    global_model = create_model(model_type, input_dim, random_state)
    
    # Initialize with dummy data to create parameter structure
    X_init = np.concatenate([X for X, _ in clients_data])[:100]
    y_init = np.concatenate([y for _, y in clients_data])[:100]
    global_model.fit(X_init, y_init)
    
    # Training history
    history = defaultdict(list)
    
    print("\n" + "="*70)
    print(f"FEDERATED TRAINING: {num_clients} Clients × {num_rounds} Rounds")
    print(f"Model: {model_type.upper()}")
    print("="*70 + "\n")
    
    # Federated training loop
    for round_idx in range(num_rounds):
        print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
        
        client_models = []
        client_weights = []
        round_metrics = []
        
        # Each client trains locally
        for client_id, (X_client, y_client) in enumerate(clients_data):
            # Check if client has enough samples for stratified split
            class_counts = np.bincount(y_client)
            min_class_count = min(class_counts)
            
            # Use stratified split only if each class has ≥2 samples
            if min_class_count >= 2 and len(y_client) >= 10:
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_client, y_client, 
                        test_size=val_split,
                        stratify=y_client,
                        random_state=random_state + round_idx
                    )
                except ValueError:
                    # Fallback: non-stratified split
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_client, y_client, 
                        test_size=val_split,
                        random_state=random_state + round_idx
                    )
            else:
                # Client too small: use all data for training, validate on same data
                X_train, y_train = X_client, y_client
                X_val, y_val = X_client, y_client
                print(f"  ⚠️  Client {client_id} too small for split, using full data")
            
            # Create local model
            local_model = create_model(model_type, input_dim, random_state)
            
            # Get global parameters if not first round
            init_params = None
            if round_idx > 0:
                init_params = get_model_parameters(global_model)
            
            # Train locally with optional initialization from global model
            local_model, metrics = train_local_model(
                local_model, X_train, y_train, X_val, y_val, init_params
            )
            
            client_models.append(local_model)
            client_weights.append(len(X_client))
            round_metrics.append(metrics)
            
            print(f"  Client {client_id}: "
                  f"Train Acc={metrics['train_acc']:.3f}, "
                  f"Val Acc={metrics['val_acc']:.3f}, "
                  f"Val F1={metrics['val_f1']:.3f}")
        
        # Aggregate models (FedAvg)
        global_params = federated_averaging(client_models, client_weights)
        global_model = set_model_parameters(global_model, global_params)
        
        # Compute average metrics
        avg_train_acc = np.mean([m['train_acc'] for m in round_metrics])
        avg_val_acc = np.mean([m['val_acc'] for m in round_metrics])
        avg_val_f1 = np.mean([m['val_f1'] for m in round_metrics])
        
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['val_f1'].append(avg_val_f1)
        
        print(f"\n  🌐 Global Model - Avg Train Acc: {avg_train_acc:.3f}, "
              f"Avg Val Acc: {avg_val_acc:.3f}, Avg Val F1: {avg_val_f1:.3f}")
    
    return {
        'global_model': global_model,
        'history': dict(history),
        'client_models': client_models
    }


# ============================================================================
# SECTION 6: EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, dataset_name="Test"):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities if available
    try:
        y_proba = model.predict_proba(X_test)
        print(f"\nPrediction distribution: {np.bincount(y_pred)}")
        print(f"Actual distribution: {np.bincount(y_test)}")
    except:
        pass
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Set Evaluation:")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(['Healthy', 'Diabetic']):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            print(f"  {class_name}: {class_acc:.4f} ({class_mask.sum()} samples)")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Healthy', 'Diabetic'],
                                zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Healthy  Diabetic")
    print(f"Actual Healthy    {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"       Diabetic   {cm[1][0]:3d}      {cm[1][1]:3d}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def plot_training_history(history, save_path='training_history.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = range(1, len(history['train_acc']) + 1)
    
    # Accuracy
    axes[0].plot(rounds, history['train_acc'], 'o-', label='Train Accuracy', linewidth=2)
    axes[0].plot(rounds, history['val_acc'], 's-', label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Federated Round', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Federated Learning: Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1-Score
    axes[1].plot(rounds, history['val_f1'], 'd-', label='Val F1-Score', 
                linewidth=2, color='green')
    axes[1].set_xlabel('Federated Round', fontsize=12)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('Federated Learning: F1-Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Training history plot saved: {save_path}")


# ============================================================================
# SECTION 7: SAVE/LOAD MODELS
# ============================================================================

def save_fl_system(global_model, client_models, history, output_dir='models'):
    """Save complete FL system."""
    Path(output_dir).mkdir(exist_ok=True)
    
    joblib.dump(global_model, f'{output_dir}/global_model.pkl')
    joblib.dump(client_models, f'{output_dir}/client_models.pkl')
    joblib.dump(history, f'{output_dir}/training_history.pkl')
    
    print(f"\n✅ FL system saved to {output_dir}/")


def load_fl_system(model_dir='models'):
    """Load FL system."""
    global_model = joblib.load(f'{model_dir}/global_model.pkl')
    client_models = joblib.load(f'{model_dir}/client_models.pkl')
    history = joblib.load(f'{model_dir}/training_history.pkl')
    
    return global_model, client_models, history


# ============================================================================
# MAIN: TRAINING PIPELINE
# ============================================================================

if __name__ == "__main__":
    from processing_pipeline import load_embeddings, assemble_features, save_preprocessing_artifacts
    
    print("="*70)
    print("DiaVoc Federated Learning Training Pipeline")
    print("="*70)
    
    # 1. Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    df = load_embeddings('male_embeddings.pkl', 'female_embeddings.pkl')
    data = assemble_features(df, fit_mode=True)
    save_preprocessing_artifacts(data['scaler'], data['pca'])
    
    X = data['X']
    y = data['y']
    
    # 2. Split into train/test (hold out test set)
    print("\n[2/5] Creating train/test split...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train_full)} | Test: {len(X_test)}")
    
    # 3. Create non-IID client partitions
    print("\n[3/5] Creating non-IID client partitions (Dirichlet α=0.5)...")
    clients_data = split_non_iid_dirichlet(
        X_train_full, y_train_full, 
        num_clients=5, 
        alpha=0.5
    )
    
    # 4. Federated training
    print("\n[4/5] Starting federated training...")
    results = train_federated(
        clients_data,
        model_type='mlp',  # Can be 'mlp', 'logistic', or 'linear_svm'
        num_rounds=10,
        val_split=0.2
    )
    
    global_model = results['global_model']
    history = results['history']
    
    # 5. Final evaluation on held-out test set
    print("\n[5/5] Final evaluation on test set...")
    test_metrics = evaluate_model(global_model, X_test, y_test, "Test")
    
    # Plot and save
    plot_training_history(history)
    save_fl_system(global_model, results['client_models'], history)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"Spec Requirement: ≥0.65 accuracy | {'✅ PASS' if test_metrics['accuracy'] >= 0.65 else '❌ FAIL'}")
    print("\n✅ All models and artifacts saved to models/")