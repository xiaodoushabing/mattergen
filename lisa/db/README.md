## General Setup

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo su - postgres
# psql - does not work (psql: error: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: No such file or directory. Is the server running locally and accepting connections on that socket?)
sudo service postgresql start
psql
CREATE DATABASE mattergen_db;
\q  # to quit psql
```

## To change password of postgres user:
```bash
sudo su - postgres
psql
ALTER USER postgres WITH PASSWORD 'your_new_password';
\q
psql "postgresql://postgres:your_new_password@localhost/mattergen_db" # let psql connect to a db
```

## Checking PostgreSQL and Database Setup
1.  **Check that PostgreSQL is running:**
    ```bash
    sudo service postgresql status
    ```
    * This command checks the status of the PostgreSQL service. You should see output indicating if it's active (running).
<br>

2.  **Database URL (for reference):**
    ```bash
    echo "Your DATABASE_URL is: postgresql://postgres:password@localhost/mattergen_db"
    ```
    * This line displays the database connection URL that your `models.py` script will use.
    * **Remember to replace `password` with your actual PostgreSQL password.**

<br>

3.  **Run `models.py` to connect and create tables:**
    ```bash
    python models.py
    ```
    * This command executes your Python script. It will attempt to connect to the PostgreSQL database specified in its `DATABASE_URL` and create the `batches` and `lattices` tables if they don't already exist.

<br>

4.  **Check the database for created tables:**

    * **Connect to PostgreSQL as the `postgres` user:**
        ```bash
        sudo -i -u postgres psql
        ```
        * This command opens the PostgreSQL interactive terminal (`psql`) as the `postgres` user. You might be prompted for the `postgres` user's password.

    * **Once inside `psql`, connect to your database:**
        ```sql
        \c mattergen_db
        ```
        * This `psql` meta-command switches the connection to the `mattergen_db` database.

    * **List the tables:**
        ```sql
        \dt
        ```
        * This `psql` meta-command displays a list of the tables in the currently connected database (`mattergen_db`). You should see `batches` and `lattices` in the output if `models.py` ran successfully.

    * **Exit `psql`:**
        ```sql
        \q
        ```
        * This `psql` meta-command closes the `psql` terminal.
<br>

**Important Notes:**

* **Replace Placeholder Password:** Remember to substitute the placeholder `"password"` in the `DATABASE_URL` with your actual PostgreSQL password in both the `echo` command and, more importantly, in your `models.py` file.
* **`sudo`:** The `sudo` command is used to execute commands with administrator privileges. You might need to enter your user password when using `sudo`.
* **Working Directory:** Ensure you are in the correct directory in your terminal when running `python models.py` (the directory containing the `models.py` file) and when running `alembic` commands (the directory containing `alembic.ini`).
* **Virtual Environment:** If you are using a virtual environment (like `.venv`), make sure it is activated before running Python commands to ensure you are using the correct dependencies.

## Updating Database Migration Scripts
* Whenever you make a change to models.py, in the terminal, run 
    ```bash
    alembic revision --autogenerate -m "<your message>"
    alembic upgrade head
    ```
* If the alembic version history is corrupted,
    1. Keep a copy of alembic.ini and env.py files
    2. Delete the entire `alembic` folder
    3. Run
        ```bash
        alembic init alembic
        alembic revision --autogenerate -m "<your message, e.g., initial version"
        alembic upgrade head
        ```
* If the tables cannot be updated (e.g., because of foreign key relations), simplest way is to re-create the database when tables are empty
    1. Enter postgres terminal
    2. List databases
        ```bash
        \l
        ```
    2. Delete the database
        ```bash
        DROP DATABASE <database_name>;
        # recreate the database
        CREATE DATABASE <database_name>;
        # connect to the new database
        \q
        psql <database_name>
        # check tables
        \dt
        ```