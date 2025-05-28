<p align="center">
  <img src="logo/matbuddy-logo.png" style="width:50%;">
</p>

MatBuddy is a full-stack web application developed to streamline material generation and analysis, specifically for use in research pipelines. It provides a user-friendly interface for interacting with **[MatterGen](https://github.com/microsoft/mattergen)** (for lattice generation) and **[MatterSim](https://github.com/microsoft/mattersim)** (for energy simulation/validation).


---

## 🏗️ Project Architecture

MatBuddy is a full-stack application with the following components:

* **Frontend**: A React application built with Vite, styled using TailwindCSS. It provides the user interface for interacting with the application's features. It communicates with the backend API.
* **Backend**: A FastAPI application that serves as the API layer. It handles business logic, interacts with the MatterGen/MatterSim tools, and communicates with the MongoDB database for data persistence.
* **Database**: A MongoDB instance used to store and retrieve lattice structures and related metadata.

The typical flow is:
1.  User interacts with the Frontend UI.
2.  Frontend makes API requests to the Backend.
3.  Backend processes the request, interacting with MatterGen, MatterSim and the MongoDB database.
4.  Backend sends a response back to the Frontend.
5.  Frontend updates the UI based on the response.

🐳 All services are designed to be containerized using Docker and orchestrated with Docker Compose.

#### A simplified overview of the project structure:

```
mattergen-app/
├── backend/                # FastAPI backend application
│   ├── core/               # Config, settings, middleware
│   ├── models/             # Pydantic models for request/response
│   ├── routes/             # API endpoint definitions
│   ├── services/           # Business logic services
│   ├── database.py         # MongoDB Database connection logic
│   ├── deps.py             # Dependency definitions
│   ├── main.py             # FastAPI app instance and root/startup events
│   ├── Dockerfile          # Dockerfile for the backend
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Example for backend .env
│   └── .env                # Backend specific environment variables
├── frontend/               # React + Vite frontend application
│   ├── public/             # Static assets
│   ├── src/                # Frontend source code
│   │   ├── App.jsx         # Main application component
│   │   ├── main.jsx        # Entry point for the React app
│   │   └── components/     # Reusable UI components
│   ├── Dockerfile          # Dockerfile for the frontend
│   ├── package.json        # Node.js dependencies and scripts
│   ├── vite.config.js      # Vite configuration
│   └── .env                # Frontend specific environment variables (VITE_*)
├── .env                    # Root environment variables for Docker Compose
├── .env.example            # Example for root .env
├── docker-compose.yml      # Docker Compose configuration file
└── README.md               # This file
```
---

## 🚀 Features

- Generate new lattice structures with MatterGen, with conditioning on magnetic properties
- View and download structure data
- Persist and retrieve results from a MongoDB database

---

## 📦 Prerequisites

- Git
- Docker or Portainer

---

## 🛠️ Setup

### 1. Clone this repository
```bash
git clone https://github.com/xiaodoushabing/mattergen.git
cd lisa/mattergen-app
```

### 2. Configure Environment
1. Copy the main `.env` files:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```
2. Adjust port settings if needed:

    - `.env`: Set `HOST_FRONTEND_PORT=5173` and `HOST_BACKEND_PORT=8080` (or use any preferred values)

    - `backend/.env`: set `MONGO_HOST=mongo` if using a bundled MongoDB or `MONGO_HOST=<your-existing-service-name>` if using Docker's or Portainer's shared network.

---

### 3. Docker Setup 🐋
The app is built to run using Docker Compose. There are two deployment scenarios:

#### Default: Portainer Environment
If you're deploying with Portainer and already have a MongoDB container:

- The compose file expects a pre-existing Docker network named `<network-name>`.
- MongoDB must be on the same network.
- You don't need to uncomment any service blocks.

Ensure in docker-compose.yml:
```yaml
networks:
  matbuddy_network:
    external: true
    name: <network-name>
```
#### Alternative: Local Standalone Deployment
If you're testing locally or don't have an existing MongoDB instance and want Docker Compose to manage a new MongoDB container for you:

1.  Navigate to `docker-compose.yml`, comment/uncomment accordingly
    
2.  **Be sure to modify the `networks` block:**
    Change the `matbuddy_network` definition at the end of the file to let Docker Compose create and manage the network.

    Comment:
    ```yaml
    # networks:
    #   matbuddy_network:
    #     external: true
    #     name: app-network
    ```
    and uncomment:
    ```yaml
    # Uncomment to define a new network for your app services
    app-network:  # Define a new network for your app services
       driver: bridge # Standard bridge network
    ```

---
###  4. Building & Running 🧱

#### Default: Using Portainer with existing Mongo container
1. Ensure mongo container is named `mongo` and running
2. Navigate to `Stacks` and `Add stack`.
3. Name your stack
4. Select `repository` as build method
5. Copy repository URL
6. If you cloned the repository and did not make any changes, your referece is main and can leave the field blank (default is refs/head/main)
7. Compose path is `./lisa/mattergen-app/docker-compose.yml`
8. Deploy the stack

#### Using Docker
From the root directory:

```bash
docker-compose up -d --build
```
⚠️ Note: First-time setup may take time — MatterSim and PyTorch dependencies are large. The backend uses an NVIDIA PyTorch container, and SHMEM allocation may be insufficient by default. You may add these flags to your docker run (if running manually):

```bash
--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
```
---
### Accessing the App 🌐
Once containers are running, open your browser:

- Frontend: `http://<ip-of-host>:5173`

- Backend API: `http://<ip-of-host>:8080`

Replace <ip-of-host> with your host's IP address (run `hostname -I` or `ip addr` on the host to get this).

> If you're using Portainer on a remote server, make sure these ports are open in your firewall/security group.

---
### Development Notes 🧑‍💻
##### TailwindCSS
I use TailwindCSS with PostCSS in the frontend. You can edit styles using utility classes directly in components. For custom styling, add to index.css or Tailwind config.

To run frontend locally (if not using Docker):

```bash
cd frontend
npm install
npm run dev
```
---
### Cleanup 🧼
To stop and remove all containers, networks, and volumes:
```bash
docker-compose down
```
To just stop services (without removing):
```bash
docker-compose stop
```
---

### Acknowledgments
- Developed at Seagate Research Group for experimental materials discovery workflows.
- Leverages MatterGen and MatterSim for generative modeling and simulation validation.

---

### ✅ Status
Deployment might take a while due to slow mattersim build times — if the frontend shows a blank page, wait a minute and refresh.
> Access the app here: `http://<ip-of-host>:5173`