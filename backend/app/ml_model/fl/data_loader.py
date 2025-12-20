import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_process_data(male_pkl_path, female_pkl_path):
    # Fallback paths to handle different directory structures
    paths = [
        (male_pkl_path, female_pkl_path),
        ('../male_embeddings.pkl', '../female_embeddings.pkl'),
        ('male_embeddings.pkl', 'female_embeddings.pkl')
    ]
    
    m_path, f_path = None, None
    for m, f in paths:
        if os.path.exists(m) and os.path.exists(f):
            m_path, f_path = m, f
            break
            
    if not m_path:
        raise FileNotFoundError("Could not find the .pkl files. Please check your paths.")

    # Load and combine
    df_m = pd.read_pickle(m_path)
    df_f = pd.read_pickle(f_path)
    df = pd.concat([df_m, df_f], ignore_index=True)

    # 1. Feature Engineering: The 'Risk Index'
    # Clinical studies show Age and BMI interact significantly for diabetes risk
    df['gender_bin'] = df['gender'].apply(lambda x: 1 if str(x).lower() == 'male' or x == 1 else 0)
    df['risk_index'] = (df['age'] * df['bmi']) / 100.0

    # 2. Extract Tabular & Voice Features
    X_tab = df[['age', 'bmi', 'gender_bin', 'risk_index']].values
    X_emb = np.stack(df['byols_embeddings'].values)

    # 3. PCA: Retain 95% variance to filter out background recording noise
    pca = PCA(n_components=0.95, random_state=42)
    X_emb_pca = pca.fit_transform(X_emb)

    # Combine: [Tabular (4) | Voice (Reduced)]
    X = np.hstack([X_tab, X_emb_pca])
    y = df['diabetes'].values.astype(int)

    # 4. Global Scaling (Essential for PCA and model stability)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"✅ Data processed successfully: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, y

def split_non_iid(X, y, num_clients=5):
    """Shuffles and splits data into shards for Federated Learning."""
    indices = np.random.permutation(len(X))
    split_indices = np.array_split(indices, num_clients)
    return [(X[idx], y[idx]) for idx in split_indices]

if __name__ == "__main__":
    # Test the loader
    try:
        # We use dummy paths because the fallback logic will find your real .pkl files
        X, y = load_and_process_data('male_embeddings.pkl', 'female_embeddings.pkl')
        
        print("\n--- Data Summary ---")
        print(f"Total Samples: {len(X)}")
        print(f"Features per sample: {X.shape[1]}")
        print(f"Diabetic cases: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"Healthy cases: {len(y) - sum(y)}")
        print("--------------------")
        
    except Exception as e:
        print(f"❌ Error during data loading: {e}")