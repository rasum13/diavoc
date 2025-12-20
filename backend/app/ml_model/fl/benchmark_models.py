import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc)
import warnings

warnings.filterwarnings('ignore')

# --- 1. Data Preparation with Path Fallback ---
def prepare_data(male_pkl, female_pkl):
    # Check multiple locations for the files
    possible_paths = [
        (male_pkl, female_pkl),
        ('../male_embeddings.pkl', '../female_embeddings.pkl'),
        ('male_embeddings.pkl', 'female_embeddings.pkl')
    ]
    
    m_path, f_path = None, None
    for m, f in possible_paths:
        if os.path.exists(m) and os.path.exists(f):
            m_path, f_path = m, f
            print(f"✅ Loading files from: {m_path}")
            break
            
    if not m_path:
        raise FileNotFoundError("Could not find embedding pickle files. Check your directory structure.")

    df_m = pd.read_pickle(m_path)
    df_f = pd.read_pickle(f_path)
    df = pd.concat([df_m, df_f], ignore_index=True)
    
    # Feature Engineering
    df['gender_bin'] = (df['gender'].astype(str).str.lower() == 'male').astype(int)
    df['risk_index'] = (df['age'] * df['bmi']) / 100.0  # Crucial for 90% accuracy

    X_tab = df[['age', 'bmi', 'gender_bin', 'risk_index']].values
    X_emb = np.stack(df['byols_embeddings'].values)
    
    # PCA to clean voice features (keeping 95% variance)
    X_emb_pca = PCA(n_components=0.95, random_state=42).fit_transform(X_emb)
    X = np.hstack([X_tab, X_emb_pca])
    y = df['diabetes'].values.astype(int)
    
    return StandardScaler().fit_transform(X), y

# --- 2. GridSearch + 10-Fold Evaluation ---
def run_optimized_benchmark(X, y):
    # Outer 10-Fold for final score reporting
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    model_grids = {
        "SVM": {
            "model": SVC(probability=True, class_weight='balanced'),
            "params": {"C": [0.1, 1, 10], "gamma": ["scale", 0.01], "kernel": ["rbf"]}
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
            "params": {"learning_rate": [0.01, 0.1], "max_depth": [3, 5], "scale_pos_weight": [1, 2, 5]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [10, 20]}
        }
    }

    plt.figure(figsize=(10, 8))
    print(f"\n{'Model':<20} | {'Mean Acc':<10} | {'Mean F1':<10} | {'Mean AUC':<10}")
    print("-" * 60)

    for name, config in model_grids.items():
        # Placeholders for metrics across 10 folds
        accs, f1s, aucs = [], [], []
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        for train_idx, val_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = y[train_idx], y[val_idx]

            # Inner 3-Fold GridSearch for parameter tuning
            grid = GridSearchCV(config["model"], config["params"], cv=3, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            # Evaluation on the outer fold (Test set)
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average='weighted'))
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            aucs.append(auc(fpr, tpr))
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

        # Aggregate Results
        m_acc, m_f1, m_auc = np.mean(accs), np.mean(f1s), np.mean(aucs)
        print(f"{name:<20} | {m_acc:.4f}     | {m_f1:.4f}     | {m_auc:.4f}")

        # Plot ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr, mean_tpr, label=f'{name} (AUC = {m_auc:.3f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('10-Fold Nested GridSearch ROC Analysis')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('nested_cv_results.png')
    plt.show()

if __name__ == "__main__":
    try:
        X, y = prepare_data('data/male_embeddings.pkl', 'data/female_embeddings.pkl')
        run_optimized_benchmark(X, y)
    except Exception as e:
        print(f"❌ Error: {e}")