from sqlalchemy import Column, Integer, String, Numeric, VARCHAR
from database import Base

class NumberOfRounds(Base):
    __tablename__ = "NFHS_Rounds"

    nfhs_id = Column(Numeric, primary_key=True, index=True)
    nfhs_round = Column(VARCHAR, index=True)
    # email = Column(String, unique=True, index=True)