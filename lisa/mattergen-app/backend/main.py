from fastapi import FastAPI
from routes import generate_lattice, retrieval

app = FastAPI(title="MatterGen & MatterSim API")

app.include_router(generate_lattice.router)
app.include_router(retrieval.router)
