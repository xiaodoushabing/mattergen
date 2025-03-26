import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY
import datetime

load_dotenv()

Base = declarative_base()

#%%
class Batch(Base):
    __tablename__ = "batches"
    id = Column(Integer, primary_key=True, autoincrement=True)
    guidance_factor = Column(Float, nullable=False)
    magnetic_density = Column(Float, nullable=False)
    file_path = Column(String, nullable=True)
    added_at = Column(DateTime, default=datetime.datetime.utcnow)

    # One-to-many relationship: a batch has multiple lattices
    # relationship = relationshio(RelatedClass, BidirectionalRelationship )
    lattices = relationship("Lattice", back_populates="batch")

class Lattice(Base):
    __tablename__ = "lattices"
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(Integer, ForeignKey("batches.id"))
    no_of_atoms = Column(Integer, nullable = False)
    cell_parameters = Column(ARRAY(Float, dimensions=1), nullable=False)  # expect 9 values in one-dimensional array
    pbc = Column(String, nullable=False)
    atoms_list=Column(JSON, nullable=False) #{Fe: 3, O: 2}
    atoms = Column(JSON, nullable=False) 
    """
    "atoms": {
    "1": {"element": "C", "coords": [0.1, 0.2, 0.3]},
    "2": {"element": "O", "coords": [0.4, 0.5, 0.6]},
    "3": {"element": "H", "coords": [0.7, 0.8, 0.9]}
    }"""
    
    dft_predicted_energy = Column(Float, nullable = True)
    dft_predicted_magnetic_property = Column(Float, nullable = True)
    mattersim_predicted_energy = Column(Float, nullable=True)

    batch = relationship("Batch", back_populates="lattices")

# Database connection configuration
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = os.environ.get("DB_PORT")

if DB_PORT is None:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
else:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print(f"Database URL: {DATABASE_URL}")