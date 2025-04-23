from fastapi import FastAPI
from routes import generate_lattice, retrieval, download
from database import connect_to_mongo
from core.middleware_config import LogRequestMiddleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MatBuddy API")

# define allowed origins
origins = [
    "http://localhost:5173", # Vite default dev server
]

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Allows requests that include credentials like cookies
    allow_methods=["*"],  # Allows all standard HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all request headers
)

# add custom logging middleware
app.add_middleware(LogRequestMiddleware)

@app.on_event("startup")
async def setup_indexes():
    _, lattice_collection =connect_to_mongo()
    lattice_collection.create_index([("guidance_factor", 1),
                                     ("magnetic_density", 1),
                                     ("no_of_atoms", 1),
                                     ("ms_predictions.energy", 1)
                                     ])
@app.get("/")
async def read_root():
    return {"message": "Welcome to MatBuddy API"}

app.include_router(generate_lattice.router)
app.include_router(retrieval.router)
app.include_router(download.router)
