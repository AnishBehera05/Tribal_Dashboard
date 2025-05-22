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
@app.get("/States/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    States_Information = db.query(models.States_Information).all()
    return States_Information

@app.get("/Districts/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    Districts_Information = db.query(models.Districts_Information).all()
    return Districts_Information

@app.get("/Categories/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    Categories_Information = db.query(models.Categories_Information).all()
    return Categories_Information

@app.get("/Indicators/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    Indicators_Information = db.query(models.Indicators_Information).all()
    return Indicators_Information

@app.get("/NFHS_Rounds/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    NFHS_Information = db.query(models.NFHS_Information).all()
    return NFHS_Information

@app.get("/NFHS_State_Data/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    NFHS_State_Data_Information = db.query(models.NFHS_State_Data_Information).all()
    return NFHS_State_Data_Information

@app.get("/NFHS_District_Data/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    NFHS_District_Data_Information = db.query(models.NFHS_District_Data_Information).all()
    return NFHS_District_Data_Information