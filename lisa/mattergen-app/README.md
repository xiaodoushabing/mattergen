<p align="center">
  <img src="logo/matbuddy-logo.png" width="50%">
</p>

MatBuddy is a full-stack web application that streamlines material generation and analysis workflows. It provides a user-friendly interface for interacting with **[MatterGen](https://github.com/microsoft/mattergen)** (lattice generation) and **[MatterSim](https://github.com/microsoft/mattersim)** (energy simulation/validation).


---

## 🏗️ Project Architecture

#### MatBuddy consists of:
* **Frontend**: React + Vite app styled with TailwindCSS. Interfaces with the backend via REST APIs.
* **Backend**: FastAPI application that manages business logic and coordinates with MatterGen, MatterSim, and MongoDB.
* **Database**: MongoDB instance for persisting generated lattice data and metadata.

#### Workflow
1.  User interacts with the frontend UI.
2.  Frontend sends API requests to the backend.
3.  Backend processes requests and interacts with MatterGen, MatterSim, and MongoDB.
4.  Backend returns data to the frontend.
5.  Frontend updates the UI.

🐳 All components are containerized with Docker and orchestrated using Docker Compose.

#### A simplified overview of the project structure:

```
mattergen-app/
├── backend/                # FastAPI backend application
│   ├── core/               # Config, settings, middleware
│   ├── models/             # Pydantic schemas
│   ├── routes/             # API routes
│   ├── services/           # Business logic
│   ├── database.py         # MongoDB connection
│   ├── deps.py             # Dependency definitions
│   ├── main.py             # FastAPI app
│   ├── Dockerfile          
│   ├── requirements.txt    
│   ├── .env.example       
│   └── .env              
├── frontend/               # React + Vite frontend application
│   ├── public/             
│   ├── src/                
│   │   ├── App.jsx         
│   │   ├── main.jsx        
│   │   └── components/     
│   ├── Dockerfile          
│   ├── package.json        
│   ├── vite.config.js      
│   └── .env                
├── .env                    # Root .env for Docker Compose
├── .env.example            # Example for root .env
├── docker-compose.yml      # Docker Compose configuration file
└── README.md               # This file
```
---

## 🚀 Features

- Generate lattice structures with MatterGen (supports magnetic property conditioning)
- View and download structure data
- Store and retrieve results from MongoDB

---

## 📦 Prerequisites

- Git
- Docker or Portainer

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/xiaodoushabing/mattergen.git
cd lisa/mattergen-app
```

### 2. Environment Configuration
Copy and edit the `.env` files:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```

Then update:
- `.env`: e.g. `HOST_FRONTEND_PORT=5173`,  `HOST_BACKEND_PORT=8080`
 - `backend/.env`: use `MONGO_HOST=mongo` if using a bundled MongoDB or `MONGO_HOST=<your-existing-service-name>` if using custom host.

### 3. Docker Configuration 🐋
MatBuddy can be deployed in two ways:
##### Option 1: Deploy via Portainer (Recommended for Shared Environments)
- **Use This When:** You have a shared Docker environment with a pre-existing MongoDB container, running on Portainer.
- **Notes:**
    - The compose file expects an external Docker network named `<your-network-name>`.
    - Your MongoDB container must be connected to that network.
- **Ensure Your `docker-compose.yml` Includes:**
```yaml
networks:
  matbuddy_network:
    external: true
    name: <network-name>
```
#### Option 2: Local Standalone Setup via Docker or Portainer
- **Use This When:** You’re managing the deployment but want Docker Compose to manage a new MongoDB container.
<br>

- **Configuration Changes:**
1. **MongoDB Service:** Uncomment the MongoDB service definition in `docker-compose.yml`.
2. **Dependencies:** Under the `backend` service, uncomment the `depends_on` block for MongoDB.
3. **Network:** Modify the network definition so that Docker Compose creates a new network rather than using an external one.

    Comment this:
    ```yaml
    # networks:
    #   matbuddy_network:
    #     external: true
    #     name: app-network
    ```
    and uncomment this:
    ```yaml
    # Uncomment to define a new network for your app services
    networks:
        app-network:  # Define a new network for your app services
            driver: bridge # Standard bridge network
    ```
#### Note:
The `docker-compose.yml` uses the following port mapping format:
```yaml
ports:
- target: ${CONTAINER_FRONTEND_PORT}
    published: ${HOST_FRONTEND_PORT}
    protocol: tcp
```
This syntax is supported by Docker Compose (v3 and higher) and works identically whether you deploy via Portainer or directly via the Docker CLI. It maps the container’s internal port (`CONTAINER_FRONTEND_PORT`) to the host port (`HOST_FRONTEND_PORT`).

###  4. Building & Running 🧱

#### Option 1: Portainer Stack Deployment
1. Ensure a MongoDB container named `mongo` is running on the same network
2. Go to Portainer → Stacks → Add Stack.
3. Select `repository` as build method
4. Set repository URL
6. If you cloned the repository and did not make any changes, your referece is main and can leave the field blank
(default is refs/head/main)
7. Compose path: `./lisa/mattergen-app/docker-compose.yml`
8. Deploy

#### Option 2: Docker CLI
From the root directory:

```bash
docker-compose up -d --build
```
⚠️ Note: First-time builds may take several minutes as MatterSim and MatterGen dependencies are large. MatterSim requires PyTorch with GPU support. Use the following Docker flags if building manually:

```bash
--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
```
---
## Accessing the App 🌐
Once containers are running, open your browser:

- Frontend: `http://<ip-of-host>:5173`

- Backend API: `http://<ip-of-host>:8080`

> Use `hostname -I` or `ip addr` to get the IP address. Ensure firewall rules allow access to these ports.

---

## Development Notes 🧑‍💻

##### TailwindCSS
- Tailwind is used for styling via utility classes.
- Custom styles can be added in `index.css` or `tailwind.config.js`.

**Local Frontend Dev (No Docker)**

```bash
cd frontend
npm install
npm run dev
```
---
## Cleanup 🧼
To stop and remove all containers, networks, and volumes:
```bash
docker-compose down
```
To stop services without removing:
```bash
docker-compose stop
```
---

## Acknowledgments
- Developed at Seagate Research Group for experimental materials discovery.
- Powered by MatterGen and MatterSim for generative modeling and simulation.

---

## Status ✅
> MatterSim build time is long — if the frontend appears blank, wait a minute and refresh.

Access the app: `http://<host-ip>:5173`