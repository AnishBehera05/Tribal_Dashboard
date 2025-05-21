from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import models
from database import engine, get_db

# Create the database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Pydantic model for response
from pydantic import BaseModel

class UserResponse(BaseModel):
    nfhs_id: int
    nfhs_round: str

    class Config:
        orm_mode = True

# Get all users
@app.get("/NFHS_Rounds/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    NumberOfRounds = db.query(models.NumberOfRounds).all()
    return NumberOfRounds