import numpy as np
import joblib
import torch
import librosa
from pathlib import Path
import warnings
from serab_byols import load_model, get_scene_embeddings

warnings.filterwarnings('ignore')
current_dir = Path(__file__).resolve().parent

class DiaVocInferenceSystem:
    def __init__(self, model_dir=None, audio_checkpoint=None):
        # Set base_path to ml_model directory (parent of app)
        base_path = current_dir.parent
        
        # Handle model directory
        if model_dir is None or str(model_dir) == 'models':
            self.model_dir = base_path / "models"
        else:
            self.model_dir = Path(model_dir)
            # If model_dir is relative, make it relative to base_path
            if not self.model_dir.is_absolute():
                self.model_dir = (base_path / self.model_dir).resolve()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load preprocessing components
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        self.pca = joblib.load(self.model_dir / 'pca.pkl')
        self.feature_selector = joblib.load(self.model_dir / 'feature_selector.pkl')
        
        # Load ML model
        model_path = self.model_dir / 'global_model_improved.pkl'
        self.model = joblib.load(model_path)
        
        # Handle audio checkpoint path
        if audio_checkpoint is None:
            # Default checkpoint path
            checkpoint_file = "default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth"
            audio_checkpoint = base_path / "serab-byols" / "checkpoints" / checkpoint_file
        else:
            # Convert to Path object
            audio_checkpoint = Path(audio_checkpoint)
            
            # If it's not absolute, resolve it relative to base_path
            if not audio_checkpoint.is_absolute():
                # Join with base_path and resolve to absolute path
                audio_checkpoint = (base_path / audio_checkpoint).resolve()
        
        # Verify checkpoint exists before trying to load
        if not audio_checkpoint.exists():
            raise FileNotFoundError(
                f"Audio checkpoint not found!\n"
                f"Expected at: {audio_checkpoint}\n"
                f"Base path: {base_path}\n"
                f"File exists: {audio_checkpoint.exists()}\n"
                f"Please verify the checkpoint file is in the correct location."
            )
        
        print(f"Loading audio checkpoint from: {audio_checkpoint}")
        self.audio_encoder = load_model(str(audio_checkpoint))
        self.audio_encoder.to(self.device)
        self.audio_encoder.eval()
        
        self.sample_rate = 16000
    
    def _extract_voice_embedding(self, wav_path):
        """Extract voice embedding from audio file."""
        audio, _ = librosa.load(wav_path, sr=self.sample_rate)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = get_scene_embeddings(audio_tensor, self.audio_encoder)
        
        return embedding.cpu().numpy().flatten()
    
    def _extract_handcrafted(self, embedding):
        """Extract handcrafted features from embedding."""
        chunk_size = len(embedding) // 16
        means = [np.mean(embedding[i*chunk_size:(i+1)*chunk_size]) for i in range(16)]
        stds = [np.std(embedding[i*chunk_size:(i+1)*chunk_size]) for i in range(16)]
        return np.array(means + stds)
    
    def predict_from_wav(self, wav_path, age, gender, bmi, ethnicity='asian'):
        """
        Predict diabetes risk from voice recording and patient data.
        
        Args:
            wav_path: Path to audio file
            age: Patient age
            gender: Patient gender ('male' or 'female')
            bmi: Patient BMI
            ethnicity: Patient ethnicity (default 'asian')
            
        Returns:
            dict with 'diagnosis' and 'probability' keys
        """
        # Extract voice embedding
        raw_embedding = self._extract_voice_embedding(wav_path)
        
        # Process inputs
        gender_bin = 1 if str(gender).lower() in ['male', '1'] else 0
        ethnicity_bin = 1 if str(ethnicity).lower() in ['asian', 'asia'] else 0
        tabular = np.array([age, gender_bin, bmi, ethnicity_bin])
        
        # Extract handcrafted features
        handcrafted = self._extract_handcrafted(raw_embedding)
        
        # Reduce embedding dimensions
        reduced_emb = self.pca.transform(raw_embedding.reshape(1, -1))
        
        # Combine all features
        X = np.hstack([
            tabular.reshape(1, -1),
            handcrafted.reshape(1, -1),
            reduced_emb
        ])
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_final = self.feature_selector.transform(X_scaled)
        
        # Predict
        prob = self.model.predict_proba(X_final)[0][1]
        prediction = int(prob >= 0.5)
        
        return {
            "diagnosis": "DIABETIC" if prediction else "HEALTHY",
            "probability": float(prob)
        }
