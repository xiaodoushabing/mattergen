from fastapi import FastAPI
from routes import generate_lattice, retrieval
from database import connect_to_mongo

app = FastAPI(title="MatterGen & MatterSim API")

@app.on_event("startup")
def setup_indexes():
    _, lattice_collection =connect_to_mongo()
    lattice_collection.create_index([("guidance_factor", 1),
                                     ("magnetic_density", 1),
                                     ("no_of_atoms", 1),
                                     ("ms_predictions.energy", 1)
                                     ])
@app.get("/")
def read_root():
    return {"message": "FastAPI with MongoDB Indexing"}

app.include_router(generate_lattice.router)
app.include_router(retrieval.router)
