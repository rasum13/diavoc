import io
from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse
from app.api.deps import get_current_user
from app.crud.history import create_history_item
from app.db.session import get_db
from app.db.models.screening_history import ScreeningHistory
from app.db.models.user import User
from app.schemas.history import HistoryItem, HistoryItemCreate, ScreeningHistoryOutput
from app.schemas.user import UserOutput

router = APIRouter()

# history_list = [
#     {"date": "2025-09-10", "score": 0.25, "accuracy": 0.71},
#     {"date": "2025-09-13", "score": 0.41, "accuracy": 0.74},
#     {"date": "2025-09-17", "score": 0.45, "accuracy": 0.63},
#     {"date": "2025-09-21", "score": 0.71, "accuracy": 0.78},
# ]
#

@router.get("/")
def get_history(limit: int | None = None, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    history_list = db.query(ScreeningHistory).filter(ScreeningHistory.user_id == current_user.id).all()
    # print(history_list)
    if limit:
        return history_list[:limit]
    return history_list


@router.post("/add")
def add_history(history_item: HistoryItemCreate, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    new_history = create_history_item(db, history_item, user_id=current_user.id)
    return new_history


@router.get("/{screening_id}", response_model=ScreeningHistoryOutput)
def get_screening(
    screening_id: int,
    current_user: UserOutput = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific screening record.
    """
    screening = db.query(ScreeningHistory).filter(
        ScreeningHistory.id == screening_id,
        ScreeningHistory.user_id == current_user.id
    ).first()
    
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    
    return screening

@router.get("/{screening_id}/waterfall-plot")
def get_waterfall_plot(
    screening_id: int,
    current_user: UserOutput = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the waterfall plot image for a specific screening.
    """
    screening = db.query(ScreeningHistory).filter(
        ScreeningHistory.id == screening_id,
        ScreeningHistory.user_id == current_user.id
    ).first()
    
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    
    if not screening.waterfall_plot:
        raise HTTPException(status_code=404, detail="Waterfall plot not available for this screening")
    
    return StreamingResponse(
        io.BytesIO(screening.waterfall_plot),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=waterfall_{screening_id}.png"}
    )

@router.get("/{screening_id}/force-plot")
def get_force_plot(
    screening_id: int,
    current_user: UserOutput = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the force plot image for a specific screening.
    """
    screening = db.query(ScreeningHistory).filter(
        ScreeningHistory.id == screening_id,
        ScreeningHistory.user_id == current_user.id
    ).first()
    
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    
    if not screening.force_plot:
        raise HTTPException(status_code=404, detail="Force plot not available for this screening")
    
    return StreamingResponse(
        io.BytesIO(screening.force_plot),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=force_{screening_id}.png"}
    )

@router.delete("/{screening_id}")
def delete_screening(
    screening_id: int,
    current_user: UserOutput = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a screening record.
    """
    screening = db.query(ScreeningHistory).filter(
        ScreeningHistory.id == screening_id,
        ScreeningHistory.user_id == current_user.id
    ).first()
    
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    
    db.delete(screening)
    db.commit()
    
    return {"message": "Screening deleted successfully"}
