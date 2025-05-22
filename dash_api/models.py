from sqlalchemy import Column, Integer, String, Numeric, VARCHAR
from database import Base

class States_Information(Base):
    __tablename__ = "States"

    state_id = Column(Numeric, primary_key=True, index=True)
    state_name = Column(VARCHAR, index=True)
    state_acronym = Column(VARCHAR, index=True)

class Districts_Information(Base):
    __tablename__ = "Districts"

    district_id = Column(Numeric, primary_key=True, index=True)
    state_id = Column(Numeric, index=True)
    district_name = Column(VARCHAR, index=True)

class Categories_Information(Base):
    __tablename__ = "Categories"

    categories_id = Column(Numeric, primary_key=True, index=True)
    categories = Column(VARCHAR, index=True)

class Indicators_Information(Base):
    __tablename__ = "Indicators"

    indicators_id = Column(Numeric, primary_key=True, index=True)
    indicator_name = Column(VARCHAR, index=True)
    indicators_type_id = Column(Numeric, index=True)
    indicator_type = Column(VARCHAR, index=True)

class NFHS_Rounds_Information(Base):
    __tablename__ = "NFHS_Rounds"

    nfhs_id = Column(Numeric, primary_key=True, index=True)
    nfhs_round = Column(VARCHAR, index=True)

class NFHS_State_Data_Information(Base):
    __tablename__ = "NFHS_State_Data"

    data_id = Column(Numeric, primary_key=True, index=True)
    state_id = Column(Numeric, index=True)
    indicator_id = Column(Numeric, index=True)
    categories_id = Column(Numeric, index=True)
    nfhs_id = Column(Numeric, index=True)
    st = Column(VARCHAR, index=True)
    non_st = Column(VARCHAR, index=True)
    total = Column(VARCHAR, index=True)

class NFHS_District_Data_Information(Base):
    __tablename__ = "NFHS_District_Data"

    data_id = Column(Numeric, primary_key=True, index=True)
    state_id = Column(Numeric, index=True)
    district_id = Column(Numeric, index=True)
    indicator_id = Column(Numeric, index=True)
    categories_id = Column(Numeric, index=True)
    nfhs_id = Column(Numeric, index=True)
    st = Column(VARCHAR, index=True)
    non_st = Column(VARCHAR, index=True)
    total = Column(VARCHAR, index=True)