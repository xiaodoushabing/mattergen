#%%
from pymongo import MongoClient
import os, sys
from dotenv import load_dotenv
import re
from datetime import datetime

# os.path.join(os.path.dirname(__file__), "/utils")
from utils.parser import parse_extxyz_to_json
from utils.mattersim import mattersim_prediction

## ensure files are stored in the format: dft_mag_density_<magnetic density>_<guidance factor>
#%%
load_dotenv()
mongo_host = os.environ.get("MONGO_HOST")
mongo_port = os.environ.get("MONGO_PORT")
db_name = os.environ.get("DB_NAME")
lattice_name = os.environ.get("LATTICE_NAME")

#%% define function to add data into db
def add_data(model_results_dir, cal_mattersim=True, model=5):
    """
    The function processes model results, extracts necessary data, and stores it in a MongoDB database. 
    Specifically, it:
        1. Extracts the magnetic density and guidance factor from the model results directory name.
        2. Creates a Lattice document and stores it in the MongoDB database.
        3. Parses the .extxyz file containing lattice data and stores each lattice as a separate document in MongoDB.
        4. If `cal_mattersim` is True, runs energy predictions using the specified model (default is model 5) 
           and stores the predictions along with the lattice data in MongoDB.
    
    Args:
        model_results_dir (str): The directory containing the model results. 
                                  It should include the magnetic density and 
                                  guidance factor in its name (e.g., "mag_density_guidance_factor").
        cal_mattersim (bool, optional): If True, MatterSim energy predictions will be run and stored. 
                                        Defaults to True.
        model (int, optional): The model used for MatterSim predictions (default is 5). 
                               This corresponds to the specific model version used for predictions.
    
    Returns:
        None: This function does not return any value. It directly inserts the lattice and 
              prediction data into MongoDB for each lattice.
    
    Raises:
        Exception: If any errors occur during the extraction of values, parsing of the .extxyz file, 
                   or insertion into MongoDB, an exception will be raised, and the program will exit.
    """

    # connect to MongoDB
    print(f"Connecting to: MONGO_HOST: {mongo_host}, MONGO_PORT: {mongo_port}, DB_NAME: {db_name}, LATTICE_NAME: {lattice_name}")
    client = MongoClient(f"mongodb://{mongo_host}:{int(mongo_port)}/")
    db = client[db_name]
    lattice_collection = db[lattice_name]

    # extract magnetic density and guidance factor
    try:
        match = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)$', model_results_dir.rstrip("/"))

        if not match:
            print(f"Magnetic density and guidance factor not extracted from directory name: {model_results_dir}")
            raise
        
        try:
            magnetic_density = float(match.group(1))
            guidance_factor = float(match.group(2))
            print(f"Magnetic Density: {magnetic_density}")
            print(f"Guidance Factor: {guidance_factor}")
        except ValueError as e:
            print(f"ERROR: Could not convert extracted values to float. {e}")
            raise

        # parse .extxyz into structured JSON
        structures_path = f"{model_results_dir}generated_crystals.extxyz"

        #only parse lattice data, don't do any energy predictions
        lattices_data = parse_extxyz_to_json(structures_path)

        if not lattices_data:
            print(f"ERROR: Failed to parse .extxyz file at {structures_path}")
            raise

        if cal_mattersim:
            print(f"Starting Mattersim predictions...")
            try: 
                ms_predictions = mattersim_prediction(structures_path, model=model)
                print(f"Mattersim predictions completed.")
            except Exception as e:
                print(f"ERROR: Failed to run MatterSim predictions. Error: {e}")
                print("Proceeding without Mattersim predictions")
                ms_predictions = None

        # insert lattices into MongoDB
        for (index, lattice) in enumerate(lattices_data, start=1):
            lattice_doc = {
                "lattice_index": index,
                "guidance_factor": guidance_factor,
                "magnetic_density": magnetic_density,
                "file_path": model_results_dir,
                "no_of_atoms": lattice["no_of_atoms"],
                "cell_parameters": lattice["cell_params"],
                "pbc": lattice["pbc"],
                "atoms_list": lattice["atoms_list"],  # JSON: {"Fe": 2, "O": 3}
                "atoms": {str(k): v for k, v in lattice["atoms"].items()}
            }

            if cal_mattersim and ms_predictions:
                lattice_doc["ms_predictions"] = {
                    "energy": ms_predictions[0][index-1],
                    "forces": ms_predictions[1][index-1].tolist(),
                    "stresses": ms_predictions[2][index-1].tolist()
                }
            try:
                # print(f"Lattice document to insert: {lattice_doc}")

                lattice_collection.insert_one(lattice_doc)
                print(f"Successfully inserted lattice {index} for {model_results_dir}")
            except Exception as e:
                print(f"ERROR: Failed to insert lattice {index} into MongoDB for {model_results_dir}: {e}")
                raise
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    finally:
        # close the MongoDB connection
        client.close()
        print("MongoDB connection closed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <model_results_directory>")
        # sample model_results_directory: ../results/dft_mag_density_3_3/
        sys.exit(1)

    model_results_dir = sys.argv[1]
    try:
        add_data(model_results_dir)
    except Exception as e:
        print(f"ERROR: An error occurred during data processing: {e}")
        sys.exit(1)
    sys.exit(0)