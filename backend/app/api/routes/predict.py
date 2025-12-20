from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import uuid
from pathlib import Path
from pydub import AudioSegment
import io
from datetime import datetime

from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.crud.user import get_user
from app.db.session import get_db
from app.db.models.screening_history import ScreeningHistory
from app.ml_model.app.inference import DiaVocInferenceSystem
from app.ml_model.app.xai_service import XAIService
from app.schemas.user import UserOutput

# Load model ONCE at startup
diavoc = DiaVocInferenceSystem(
    model_dir="models",
    audio_checkpoint="serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth",
)

# Initialize XAI service
try:
    xai_service = XAIService(model_dir="models")
    xai_enabled = True
    print("XAI service initialized successfully")
except Exception as e:
    print(f"Warning: XAI service initialization failed: {e}")
    xai_service = None
    xai_enabled = False

UPLOAD_DIR = Path("temp_audio")
UPLOAD_DIR.mkdir(exist_ok=True)

router = APIRouter()

@router.post("/")
def predict(
    audio: UploadFile = File(...),
    current_user: UserOutput = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict diabetes risk from voice recording.
    Now includes XAI explanations stored in the database.
    """
    file_id = f"{uuid.uuid4()}.wav"
    file_path = UPLOAD_DIR / file_id
    user_db = get_user(db, current_user.id)

    try:
        # Save uploaded audio to temporary file
        audio_bytes = audio.file.read()
        audio_mem = io.BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_mem)
        audio_segment.export(file_path, format="wav")

        # Get prediction with features for XAI
        result = diavoc.predict_with_features(
            wav_path=str(file_path),
            age=user_db.age,
            gender=user_db.gender,
            bmi=user_db.bmi,
            ethnicity="asian" if user_db.is_asian else "nonasian",
        )
        
        # Extract features and basic result
        X_final = result.pop("features")
        
        # Generate XAI plots
        waterfall_bytes = None
        force_bytes = None
        
        if xai_enabled and xai_service is not None:
            try:
                waterfall_bytes, force_bytes = xai_service.generate_explanations(X_final)
                print("XAI plots generated successfully")
            except Exception as e:
                print(f"Warning: Failed to generate XAI plots: {e}")
        
        # Save to database
        screening_record = ScreeningHistory(
            user_id=current_user.id,
            score=result["probability"],  # Store probability as score
            date=datetime.utcnow(),
            waterfall_plot=waterfall_bytes,
            force_plot=force_bytes
        )
        
        db.add(screening_record)
        db.commit()
        db.refresh(screening_record)
        
        # Return result with screening ID
        return {
            **result,
            "screening_id": screening_record.id,
            "date": screening_record.date.isoformat(),
            "xai_available": waterfall_bytes is not None
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")
    finally:
        # Clean up temporary file
        file_path.unlink(missing_ok=True)
