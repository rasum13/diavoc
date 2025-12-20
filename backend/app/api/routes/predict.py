from fastapi import APIRouter, Depends, UploadFile, File, Form
import shutil
import uuid
from pathlib import Path
from pydub import AudioSegment
import io

from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.crud.user import get_user
from app.db.session import get_db
from app.ml_model.app.inference import DiaVocInferenceSystem
from app.schemas.user import UserOutput

# Load model ONCE at startup
diavoc = DiaVocInferenceSystem(
    model_dir="models",
    audio_checkpoint="serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth",
)

UPLOAD_DIR = Path("temp_audio")
UPLOAD_DIR.mkdir(exist_ok=True)

router = APIRouter()

@router.post("/")
def predict(
    audio: UploadFile = File(...),
    # age: int = Form(...),
    # gender: str = Form(...),
    # bmi: float = Form(...),
    # ethnicity: str = Form("asian"),
    current_user: UserOutput = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_id = f"{uuid.uuid4()}.wav"
    file_path = UPLOAD_DIR / file_id
    user_db = get_user(db, current_user.id)

    try:
        audio_bytes = audio.file.read()
        audio_mem = io.BytesIO(audio_bytes)

        audio_segment = AudioSegment.from_file(audio_mem)
        audio_segment.export(file_path, format="wav")

        result = diavoc.predict_from_wav(
            wav_path=str(file_path),
            age=user_db.age,
            gender=user_db.gender,
            bmi=user_db.bmi,
            ethnicity="asian" if user_db.is_asian else "nonasian",
        )
        return result
    except Exception as e:
        return {"error": f"Failed to process audio: {str(e)}"}
    finally:
        file_path.unlink(missing_ok=True)
