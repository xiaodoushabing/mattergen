```bash

mattergen-app/
│── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Configuration (MongoDB settings, etc.)
│   ├── models.py            # Pydantic models (Request/Response)
│   ├── database.py          # MongoDB connection logic
│   ├── services.py          # Functions for running MatterGen & MatterSim
│   ├── routes/
│   │   ├── generate.py      # Endpoint for lattice generation
│   │   ├── simulate.py      # Endpoint for MatterSim calculations
│   │   ├── results.py       # Endpoint to fetch MongoDB results
│   ├── utils.py             # Helper functions
│── frontend/                # (Later) React frontend will go here
│── .env                     # Environment variables
│── docker-compose.yml       # (Optional) For containerizing API & DB
│── requirements.txt         # Python dependencies
│── README.md
```