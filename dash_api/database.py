from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# URL-encode the password: @ -> %40, # -> %23
db_string = 'postgresql+psycopg2://postgres:Anish%402003%230506@localhost:5432/postgres'

# Create the SQLAlchemy engine
engine = create_engine(db_string)

# # Test the connection
# try:
#     with engine.connect() as connection:
#         result = connection.execute(text("SELECT version();"))
#         print("Connected to:", result.fetchone())
# except Exception as e:
#     print("Connection failed:", e)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()