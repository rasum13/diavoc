import torch
import librosa
import numpy as np
from serab_byols import load_model, get_scene_embeddings

class VoiceEncoder:
    def __init__(self, model_name="vggish", device=None):
        """
        Initializes the BYOL-S model.
        Common model_name options: 'vggish', 'trill'
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing BYOL-S encoder on {self.device}...")
        # load_model returns the pre-trained BYOL-S encoder
        self.model = load_model(model_name, device=self.device)
        self.sample_rate = 16000 # Standard for BYOL-S

    def process_audio(self, wav_path):
        """
        Loads wav, resamples, and generates a fixed-length embedding.
        """
        # 1. Load and Resample
        audio, sr = librosa.load(wav_path, sr=self.sample_rate)
        
        # 2. Convert to tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        
        # 3. Generate Embedding
        # get_scene_embeddings provides a single vector for the entire clip
        with torch.no_grad():
            embedding = get_scene_embeddings(audio_tensor, self.model)
        
        # Return as numpy array (removing batch dim)
        return embedding.cpu().numpy().flatten()

# Usage Example:
# encoder = VoiceEncoder()
# emb = encoder.process_audio("patient_voice.wav")