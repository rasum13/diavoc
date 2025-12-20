"""
DiaVoc Preprocessing Pipeline
Processes pre-computed BYOL embeddings + demographics → 804-dimensional feature vector
Adapted for: Pre-computed embeddings from .pkl files (no raw audio)
Spec compliant: Section 2.2, 6, 7
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_embeddings(male_pkl='male_embeddings.pkl', female_pkl='female_embeddings.pkl'):
    """
    Load pre-computed BYOL embeddings from pickle files.
    
    Expected DataFrame columns:
        - age: numeric
        - gender: categorical ('male'/'female' or 1/0)
        - bmi: numeric
        - ethnicity: categorical
        - byols_embeddings: array-like (pre-computed voice features)
        - diabetes: binary label (0/1)
    
    Returns:
        pd.DataFrame: Combined dataset
    """
    # Load both files
    df_male = pd.read_pickle(male_pkl)
    df_female = pd.read_pickle(female_pkl)
    
    # Combine
    df = pd.concat([df_male, df_female], ignore_index=True)
    
    print(f"✅ Loaded {len(df)} samples ({len(df_male)} male, {len(df_female)} female)")
    
    return df


# ============================================================================
# SECTION 2: TABULAR FEATURE ENGINEERING (Spec Section 2.2 + 7)
# ============================================================================

def extract_tabular_features(df):
    """
    Create 4 tabular features from demographics.
    
    Features:
        1. age (numeric, will be z-scored)
        2. gender (binary: male=1, female=0)
        3. bmi (numeric, will be z-scored)
        4. ethnicity (binary: asian=1, non-asian=0)
    
    Returns:
        np.ndarray: Shape (N, 4)
    """
    # Gender encoding
    df['gender_bin'] = df['gender'].apply(
        lambda x: 1 if str(x).lower() == 'male' or x == 1 else 0
    )
    
    # Ethnicity encoding (adjust based on your data)
    # Assuming ethnicity column exists; if not, default to 0
    if 'ethnicity' in df.columns:
        df['ethnicity_bin'] = df['ethnicity'].apply(
            lambda x: 1 if str(x).lower() in ['asian', 'asia'] else 0
        )
    else:
        df['ethnicity_bin'] = 0
        print("⚠️  No ethnicity column found, defaulting to 0")
    
    # Extract as array
    X_tabular = df[['age', 'gender_bin', 'bmi', 'ethnicity_bin']].values
    
    return X_tabular


# ============================================================================
# SECTION 3: HANDCRAFTED FEATURES (Spec Section 6)
# ============================================================================

def extract_handcrafted_features_from_embeddings(embeddings):
    """
    Since we don't have raw audio, we extract statistical features from 
    the BYOL embeddings themselves to simulate handcrafted features.
    
    Creates 32 features:
        - Mean of embedding chunks (16 features)
        - Std of embedding chunks (16 features)
    
    This is a workaround when audio is unavailable.
    
    Args:
        embeddings: np.ndarray, shape (N, embedding_dim)
    
    Returns:
        np.ndarray: Shape (N, 32)
    """
    N = embeddings.shape[0]
    handcrafted = np.zeros((N, 32))
    
    for i in range(N):
        emb = embeddings[i]
        
        # Split embedding into 16 chunks and compute statistics
        chunk_size = len(emb) // 16
        
        means = []
        stds = []
        
        for j in range(16):
            start_idx = j * chunk_size
            end_idx = start_idx + chunk_size
            chunk = emb[start_idx:end_idx]
            
            means.append(np.mean(chunk))
            stds.append(np.std(chunk))
        
        handcrafted[i] = np.concatenate([means, stds])
    
    return handcrafted


# ============================================================================
# SECTION 4: BYOL EMBEDDING PROCESSING (Spec Section 5 + 7)
# ============================================================================

def process_byol_embeddings(embeddings, n_components=768, fit_pca=True, pca_model=None):
    """
    Process BYOL embeddings with PCA dimensionality reduction.
    
    Args:
        embeddings: np.ndarray or list of arrays
        n_components: Target dimension (default 768 per spec)
        fit_pca: If True, fit new PCA; else use provided pca_model
        pca_model: Pre-fitted PCA (for inference)
    
    Returns:
        tuple: (X_reduced, pca_model)
            - X_reduced: np.ndarray shape (N, target_dim)
            - pca_model: Fitted PCA transformer
    """
    # Convert to numpy array if needed
    if isinstance(embeddings, pd.Series):
        embeddings = np.stack(embeddings.values)
    elif isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    n_samples, n_features = embeddings.shape
    print(f"Original embedding: {n_samples} samples × {n_features} features")
    
    # Apply PCA reduction
    if fit_pca:
        # Determine max possible components
        max_components = min(n_samples, n_features)
        
        # If original is larger than target, reduce
        if n_features > n_components and max_components >= n_components:
            pca_model = PCA(n_components=n_components, random_state=42)
            X_reduced = pca_model.fit_transform(embeddings)
            variance_explained = pca_model.explained_variance_ratio_.sum()
            print(f"✅ PCA reduced to {n_components}D (variance retained: {variance_explained:.2%})")
        
        # If we have fewer samples than target, use 95% variance instead
        elif max_components < n_components:
            print(f"⚠️  Cannot reduce to {n_components}D with only {n_samples} samples")
            print(f"    Using PCA with 95% variance retention instead...")
            pca_model = PCA(n_components=0.95, random_state=42)
            X_reduced = pca_model.fit_transform(embeddings)
            actual_components = X_reduced.shape[1]
            variance_explained = pca_model.explained_variance_ratio_.sum()
            print(f"✅ PCA reduced to {actual_components}D (variance retained: {variance_explained:.2%})")
        
        # Already smaller than target
        else:
            pca_model = None
            X_reduced = embeddings
            print(f"⚠️  Embeddings already {n_features}D (< {n_components}D target), no PCA needed")
    else:
        # Use existing PCA
        if pca_model is not None:
            X_reduced = pca_model.transform(embeddings)
        else:
            X_reduced = embeddings
    
    return X_reduced, pca_model


# ============================================================================
# SECTION 5: COMPLETE FEATURE ASSEMBLY (Spec Section 7)
# ============================================================================

def assemble_features(df, fit_mode=True, scaler=None, pca=None):
    """
    Assemble complete 804-D feature vector per spec:
        [4 tabular | 32 handcrafted | 768 BYOL] = 804 features
    
    Args:
        df: DataFrame with embeddings and demographics
        fit_mode: If True, fit scaler/PCA; if False, use provided ones
        scaler: Pre-fitted StandardScaler (for inference)
        pca: Pre-fitted PCA (for inference)
    
    Returns:
        dict: {
            'X': np.ndarray (N, 804),
            'y': np.ndarray (N,),
            'scaler': StandardScaler,
            'pca': PCA
        }
    """
    # 1. Extract tabular features (4D)
    X_tabular = extract_tabular_features(df)
    
    # 2. Extract BYOL embeddings
    embeddings = np.stack(df['byols_embeddings'].values)
    
    # 3. Process embeddings → 768D
    X_byol, pca_model = process_byol_embeddings(
        embeddings, 
        n_components=768,
        fit_pca=fit_mode,
        pca_model=pca
    )
    
    # 4. Extract handcrafted features from embeddings (32D)
    X_handcrafted = extract_handcrafted_features_from_embeddings(embeddings)
    
    # 5. Concatenate: [tabular | handcrafted | BYOL]
    X_combined = np.hstack([X_tabular, X_handcrafted, X_byol])
    
    print(f"✅ Feature vector assembled: {X_combined.shape}")
    print(f"   - Tabular: {X_tabular.shape[1]}D")
    print(f"   - Handcrafted: {X_handcrafted.shape[1]}D")
    print(f"   - BYOL (PCA): {X_byol.shape[1]}D")
    print(f"   - TOTAL: {X_combined.shape[1]}D")
    
    # 6. Scale features (z-score normalization)
    if fit_mode:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        print(f"✅ Features standardized (mean=0, std=1)")
    else:
        if scaler is None:
            raise ValueError("Scaler required for inference mode")
        X_scaled = scaler.transform(X_combined)
    
    # 7. Extract labels
    y = df['diabetes'].values.astype(int)
    
    return {
        'X': X_scaled,
        'y': y,
        'scaler': scaler,
        'pca': pca_model,
        'feature_names': {
            'tabular': ['age', 'gender', 'bmi', 'ethnicity'],
            'handcrafted': [f'hc_{i}' for i in range(32)],
            'byol': [f'byol_{i}' for i in range(X_byol.shape[1])]
        }
    }


# ============================================================================
# SECTION 6: SAVE/LOAD PREPROCESSING ARTIFACTS
# ============================================================================

def save_preprocessing_artifacts(scaler, pca, output_dir='models'):
    """Save fitted scaler and PCA for inference."""
    Path(output_dir).mkdir(exist_ok=True)
    
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    if pca is not None:
        joblib.dump(pca, f'{output_dir}/pca.pkl')
    
    print(f"✅ Preprocessing artifacts saved to {output_dir}/")


def load_preprocessing_artifacts(model_dir='models'):
    """Load scaler and PCA for inference."""
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    
    pca_path = f'{model_dir}/pca.pkl'
    pca = joblib.load(pca_path) if Path(pca_path).exists() else None
    
    return scaler, pca


# ============================================================================
# SECTION 7: INFERENCE PREPROCESSING (New Patient)
# ============================================================================

def preprocess_new_patient(age, gender, bmi, ethnicity, byol_embedding, 
                           scaler, pca):
    """
    Preprocess a single new patient for inference.
    
    Args:
        age: int/float
        gender: str ('male'/'female') or int (1/0)
        bmi: float
        ethnicity: str or int
        byol_embedding: np.ndarray (pre-computed)
        scaler: Fitted StandardScaler
        pca: Fitted PCA
    
    Returns:
        np.ndarray: Shape (1, 804) ready for model.predict()
    """
    # Create mini DataFrame for consistency
    patient_df = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'ethnicity': ethnicity,
        'byols_embeddings': byol_embedding,
        'diabetes': 0  # Dummy label
    }])
    
    # Use same pipeline in inference mode
    result = assemble_features(
        patient_df,
        fit_mode=False,
        scaler=scaler,
        pca=pca
    )
    
    return result['X']  # Shape: (1, 804)


# ============================================================================
# MAIN: EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DiaVoc Preprocessing Pipeline - Training Mode")
    print("="*70)
    
    # Load data
    df = load_embeddings('male_embeddings.pkl', 'female_embeddings.pkl')
    
    # Assemble features
    data = assemble_features(df, fit_mode=True)
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(data['scaler'], data['pca'])
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Total samples: {data['X'].shape[0]}")
    print(f"Feature dimension: {data['X'].shape[1]}D (spec requires 804D)")
    print(f"Positive cases: {data['y'].sum()} ({data['y'].mean()*100:.1f}%)")
    print(f"Negative cases: {len(data['y']) - data['y'].sum()}")
    print("\n✅ Ready for model training!")
    
    # Test inference mode
    print("\n" + "="*70)
    print("Testing Inference Mode...")
    print("="*70)
    
    scaler, pca = load_preprocessing_artifacts()
    
    # Simulate new patient
    test_embedding = df.iloc[0]['byols_embeddings']
    X_new = preprocess_new_patient(
        age=45,
        gender='male',
        bmi=28.5,
        ethnicity='asian',
        byol_embedding=test_embedding,
        scaler=scaler,
        pca=pca
    )
    
    print(f"✅ New patient preprocessed: {X_new.shape}")
    print("Ready for model.predict(X_new)")