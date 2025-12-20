import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import shap
import io
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class XAIService:
    """Service for generating SHAP explanations for diabetes predictions."""
    
    def __init__(self, model_dir=None):
        """
        Initialize XAI service with the diabetes model.
        
        Args:
            model_dir: Path to directory containing model files
        """
        if model_dir is None:
            # Default to the models directory
            current_dir = Path(__file__).resolve().parent
            model_dir = current_dir.parent / "models"
        else:
            model_dir = Path(model_dir)
        
        # Load the model
        self.model = joblib.load(model_dir / 'global_model_improved.pkl')
        
        # Try to load background data if available
        # If not available, we'll use a simpler approach
        try:
            # Try to load from the project's data directory
            data_dir = current_dir.parent.parent.parent / "data"
            self.X_background = np.load(data_dir / "X_train.npy")
            print(f"Loaded background data: {self.X_background.shape}")
        except:
            print("Warning: Could not load background data. Using simplified SHAP.")
            self.X_background = None
        
        # Initialize SHAP explainer
        # Use a subset of background data for speed (100 samples)
        if self.X_background is not None:
            bg_sample = self.X_background[:100] if len(self.X_background) > 100 else self.X_background
            self.explainer = shap.KernelExplainer(self._model_predict, bg_sample)
        else:
            # If no background data, we'll use TreeExplainer if it's a tree-based model
            try:
                self.explainer = shap.TreeExplainer(self.model)
                print("Using TreeExplainer")
            except:
                print("Warning: Could not initialize SHAP explainer. XAI may not work.")
                self.explainer = None
        
        # Feature names (you may need to load these from a file)
        # For now, generating generic names based on feature count
        self.feature_names = None
    
    def _model_predict(self, x: np.ndarray) -> np.ndarray:
        """Prediction function for SHAP."""
        return self.model.predict_proba(x)[:, 1]
    
    def _get_feature_names(self, n_features: int):
        """Generate feature names if not available."""
        if self.feature_names is None or len(self.feature_names) != n_features:
            # Generic feature names
            self.feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        return self.feature_names
    
    def generate_explanations(self, features: np.ndarray) -> tuple:
        """
        Generate SHAP explanations and return two plots as bytes.
        
        Args:
            features: Feature vector from the model (after all preprocessing)
            
        Returns:
            tuple: (waterfall_plot_bytes, force_plot_bytes)
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Cannot generate explanations.")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get feature names
        feature_names = self._get_feature_names(features.shape[1])
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(features)
            
            # Handle list output from some explainers
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Take positive class
            
            # Create SHAP Explanation object
            if hasattr(self.explainer, 'expected_value'):
                base_value = self.explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            else:
                base_value = 0.5  # Default base value
            
            shap_exp = shap.Explanation(
                values=shap_values[0] if shap_values.ndim > 1 else shap_values,
                base_values=base_value,
                data=features[0],
                feature_names=feature_names
            )
            
            # Generate waterfall plot
            waterfall_bytes = self._generate_waterfall_plot(shap_exp)
            
            # Generate force plot
            force_bytes = self._generate_force_plot(shap_exp)
            
            return waterfall_bytes, force_bytes
            
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            # Return placeholder images if SHAP fails
            return self._generate_placeholder_plot("Waterfall"), self._generate_placeholder_plot("Force")
    
    def _generate_waterfall_plot(self, shap_exp) -> bytes:
        """Generate waterfall plot and return as bytes."""
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_exp, max_display=15, show=False)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            buf.seek(0)
            
            return buf.getvalue()
        except Exception as e:
            print(f"Error generating waterfall plot: {e}")
            plt.close()
            return self._generate_placeholder_plot("Waterfall Plot Error")
    
    def _generate_force_plot(self, shap_exp) -> bytes:
        """Generate force plot and return as bytes."""
        try:
            plt.figure(figsize=(12, 3))
            shap.plots.force(shap_exp, matplotlib=True, show=False)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            buf.seek(0)
            
            return buf.getvalue()
        except Exception as e:
            print(f"Error generating force plot: {e}")
            plt.close()
            return self._generate_placeholder_plot("Force Plot Error")
    
    def _generate_placeholder_plot(self, message: str) -> bytes:
        """Generate a placeholder plot when SHAP fails."""
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, message, 
                ha='center', va='center', 
                fontsize=16, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close()
        buf.seek(0)
        
        return buf.getvalue()
