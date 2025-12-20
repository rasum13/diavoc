import numpy as np
import pandas as pd
import joblib
import torch
import librosa
from pathlib import Path
import warnings
from serab_byols import load_model, get_scene_embeddings

warnings.filterwarnings('ignore')

class DiaVocInferenceSystem:
    def __init__(self, model_dir='models', audio_checkpoint='vggish_checkpoint.pth'):
        self.model_dir = Path(model_dir)
        # Determine device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing DiaVoc Inference Engine on {self.device}...")
        
        # 1. Load ML Artifacts
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        self.pca = joblib.load(self.model_dir / 'pca.pkl')
        
        model_path = self.model_dir / 'global_model_improved.pkl'
        if not model_path.exists():
            model_path = self.model_dir / 'global_model.pkl'
        self.model = joblib.load(model_path)
        print(f"✅ ML Model & Preprocessors loaded")

        # 2. Load BYOL-S Audio Encoder
        print(f"🎬 Loading BYOL-S Audio Encoder...")
        # FIX: Removed 'device' argument and moved model manually
        self.audio_encoder = load_model(audio_checkpoint)
        self.audio_encoder.to(self.device) 
        self.audio_encoder.eval()
        self.sample_rate = 16000 
        print("✅ Audio encoder ready")

    def _extract_voice_embedding(self, wav_path):
        """Converts raw audio file to 2048-d BYOL-S embedding."""
        # Load and Resample to 16kHz
        audio, sr = librosa.load(wav_path, sr=self.sample_rate)
        
        # Convert to tensor and move to device
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        
        # Generate Scene Embedding
        with torch.no_grad():
            embedding = get_scene_embeddings(audio_tensor, self.audio_encoder)
        
        return embedding.cpu().numpy().flatten()

    def _extract_handcrafted(self, embedding):
        """Extract 32 statistical features from embedding."""
        chunk_size = len(embedding) // 16
        means = [np.mean(embedding[i*chunk_size : (i+1)*chunk_size]) for i in range(16)]
        stds = [np.std(embedding[i*chunk_size : (i+1)*chunk_size]) for i in range(16)]
        return np.array(means + stds)

    def predict_from_wav(self, wav_path, age, gender, bmi, ethnicity='asian'):
        """End-to-end prediction with feature selection."""
        # 1. Extract BYOL-S Embedding
        raw_embedding = self._extract_voice_embedding(wav_path)
        
        # 2. Process tabular data
        gender_bin = 1 if str(gender).lower() == 'male' or gender == 1 else 0
        ethnicity_bin = 1 if str(ethnicity).lower() in ['asian', 'asia'] else 0
        tabular = np.array([age, gender_bin, bmi, ethnicity_bin])
        
        # 3. Extract Handcrafted stats
        handcrafted = self._extract_handcrafted(raw_embedding)
        
        # 4. Dimensionality Reduction (PCA)
        reduced_emb = self.pca.transform(raw_embedding.reshape(1, -1))
        
        # 5. Assemble and Scale (This creates 365 features)
        X = np.hstack([
            tabular.reshape(1, -1),
            handcrafted.reshape(1, -1),
            reduced_emb
        ])
        X_scaled = self.scaler.transform(X)
        
        # 6. NEW: Apply Feature Selection (Trim 365 -> 150)
        print(f"🔍 Selecting top features (365 -> 150)...")
        # Load the selector if you haven't in __init__
        if not hasattr(self, 'feature_selector'):
            self.feature_selector = joblib.load(self.model_dir / 'feature_selector.pkl')
            
        X_final = self.feature_selector.transform(X_scaled)
        
        # 7. Predict
        prob = self.model.predict_proba(X_final)[0][1]
        prediction = 1 if prob >= 0.5 else 0
        
        return self._format_report(prediction, prob, age, gender, bmi)

    def _format_report(self, pred, prob, age, gender, bmi):
        status = "🔴 DIABETIC" if pred == 1 else "🟢 HEALTHY"
        risk = "High" if prob >= 0.85 else "Elevated" if prob >= 0.65 else "Moderate" if prob >= 0.4 else "Low"
        
        print("\n" + "="*40)
        print(f"      DIAVOC DIAGNOSTIC REPORT")
        print("="*40)
        print(f"Patient: {age}yo {gender.upper()} (BMI: {bmi})")
        print(f"Result:  {status}")
        print(f"Risk:    {risk} ({prob:.2%})")
        print("="*40 + "\n")
        
        return {"diagnosis": status, "probability": prob, "risk_level": risk}

if __name__ == "__main__":
    try:
        diavoc = DiaVocInferenceSystem(
            model_dir='models', 
            audio_checkpoint='serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth'
        )
        
        diavoc.predict_from_wav(
            wav_path='Voice 2 Diabetes  Klick Health.wav',
            age=52,
            gender='male',
            bmi=30.5
        )
    except Exception as e:
        print(f"❌ Error during inference: {e}")