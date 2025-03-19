#%%
from models import Batch, Lattice, engine
from sqlalchemy.orm import sessionmaker
import re
import sys
#%%
## ensure files are stored in the format: dft_mag_density_<magnetic density>_<guidance factor>
def _parse_extxyz_to_json(structures_path):
    """
    Parses an .extxyz file and returns structured JSON data.

    Args:
        structures_path (str): The path to the .extxyz file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a lattice structure.
              Each dictionary contains keys like 'no_of_atoms', 'cell_params', 'pbc',
              'atoms_list', and 'atoms'.
    """

    with open(structures_path, 'r') as f:
        lines = f.read().strip().split("\n")
    print(len(lines))
    lattices = []
    i=0
    idx=1

    while i < len(lines):
        atom_count = int(lines[i].strip())
        lattice = lines[i + 1].strip()
        datas = lines[i + 2:i + 2 + atom_count]

        atom_list = {}
        atoms_data = {}

        lattice_split = lattice.replace('"', '').split()
        cell_params = lattice_split[:9]
        cell_params[0] = cell_params[0].split("=")[-1]

        pbc = lattice_split[-3:]
        pbc[0] = pbc[0].split("=")[-1]
        pbc = "".join(pbc)

        # print(pbc) #TTT
        # print(cell_params) # array of 9 numbers
        # break

        for (idx, data) in enumerate(datas, start = 1):
            split_data = data.split()
            atom = split_data[0]
            # update dict of atoms
            atom_list[atom] = atom_list.get(atom, 0) + 1
            atoms_data[idx]={atom: list(map(float, split_data[1:]))
                             }
            
        lattices.append({
            "no_of_atoms": atom_count,
            "cell_params": cell_params,
            "pbc": pbc,
            "atoms_list": atom_list,
            "atoms": atoms_data,
        })

        i += 2+atom_count
        idx += 1

    return lattices
#%%
def update_batch(model_results_dir):
    """
    Extracts magnetic density and guidance factor from the directory name,
    creates a Batch object, and adds it to the database.

    Args:
        model_results_dir (str): The directory containing the model results.

    Returns:
        int: The ID of the created Batch object, or None if an error occurred.
    """
    model_results_dir = model_results_dir.rstrip("/")
    match = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)$', model_results_dir)
    if match:
        magnetic_density = float(match.group(1))
        guidance_factor = float(match.group(2))
        print(f"Magnetic Density: {magnetic_density}")
        print(f"Guidance Factor: {guidance_factor}")
    else:
        print(f"ERROR: Magnetic density and guidance factor not extracted!")
        return None
    
    # Setup SQLAlchemy session
    Session=sessionmaker(bind=engine)
    session=Session()

    db_batch=Batch(
        guidance_factor = guidance_factor,
        magnetic_density = magnetic_density,
        file_path = model_results_dir
        )
    
    session.add(db_batch)
    session.commit()
    batch_id = db_batch.id
    session.close()
    
    return batch_id
#%%
def update_lattice(model_results_dir, batch_id):
    """
    Parses an .extxyz file, creates Lattice objects, and adds them to the database.

    Args:
        model_results_dir (str): The directory containing the model results.
        batch_id (int): The ID of the Batch object to associate the lattices with.
    """

    # Parse into structured JSON

    structures_path = f"{model_results_dir}generated_crystals.extxyz"
    lattices_data = _parse_extxyz_to_json(structures_path)
    
    # Setup SQLAlchemy session
    Session = sessionmaker(bind=engine)
    session = Session()

    # insert into DB
    for lattice in lattices_data:
        db_lattice = Lattice(
            batch_id=batch_id,
            no_of_atoms=lattice["no_of_atoms"],
            cell_parameters = lattice["cell_params"],
            pbc=lattice["pbc"],
            atoms_list=lattice["atoms_list"],  # JSON: {"Fe": 2, "O": 3}
            atoms=lattice["atoms"],
        )
        session.add(db_lattice)

    session.commit()
    session.close()

def add_data(model_results_dir):
    batch_id = update_batch(model_results_dir)
    if batch_id:
        update_lattice(model_results_dir, batch_id)
    else:
        print("ERROR: Batch_id not extracted.")
    print(f"Completed processing for {model_results_dir}.")

#%%
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <model_name>")
        sys.exit(1)

    model_results_dir = sys.argv[1]
    add_data(model_results_dir)
