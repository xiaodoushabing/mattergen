#%% import libraries
import torch
from ase.io import read
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader
import re
#%% define function to extract metadata from input directory
def extract_metadata(input_dir):
    match = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)$', input_dir.rstrip("/"))
    if not match:
        raise ValueError(f"Could not parse magnetic density and guidance factor from: {input_dir}")
    
    try:
        magnetic_density = float(match.group(1))
        guidance_factor = float(match.group(2))
        print(f"Magnetic Density: {magnetic_density}, Guidance Factor: {guidance_factor}")
    except ValueError as e:
        print(f"ERROR: Could not convert extracted values to float. {e}")
        raise
    return magnetic_density, guidance_factor

#%%
def mattersim_prediction(structures_path, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running MatterSim on {device}")

    # load the model
    potential = Potential.from_checkpoint(load_path=f"MatterSim-v1.0.0-{model}M.pth", device=device)

    # load all structures from the extxyz file
    # ASE's read returns a list of Atoms objects, which can be directly passed to build_dataloader().
    structures = read(structures_path, index=":")

    # build the dataloader that is compatible with MatterSim
    dataloader = build_dataloader(structures, only_inference=True)
    # make predictions
    predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)

    return predictions

#%%
## ensure files are stored in the format: dft_mag_density_<magnetic density>_<guidance factor>
def parse_extxyz_to_json(structures_path):
    """
    Parses an .extxyz file and returns structured JSON data.

    Args:
        structures_path (str): The path to the .extxyz file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a lattice structure.
              Each dictionary contains keys like 'no_of_atoms', 'cell_params', 'pbc',
              'atoms_list', and 'atoms'.
    """

    try:
        with open(structures_path, 'r') as f:
            lines = f.read().strip().split("\n")
    except OSError as e:
        print(f"ERROR: Could not read the file '{structures_path}'. || {e}")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while processing the file {structures_path}. || {e}")
        raise

    
    lattices = []
    i=0
    idx=1

    while i < len(lines):
        try:
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

            for index, data in enumerate(datas, start=1):
                split_data = data.split()
                atom = split_data[0]
                atom_list[atom] = atom_list.get(atom, 0) + 1
                atoms_data[index] = {atom: list(map(float, split_data[1:]))}
            
            lattices.append({
                "no_of_atoms": atom_count,
                "cell_params": cell_params,
                "pbc": pbc,
                "atoms_list": atom_list,
                "atoms": atoms_data,
            })
            i += 2 + atom_count
            idx += 1
        
        except IndexError as e:
            print(f"ERROR: IndexError encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            idx += 1
            continue

        except ValueError as e:
            print(f"ERROR: ValueError encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            idx += 1
            continue

        except Exception as e:
            print(f"ERROR: Unexpected error encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            idx += 1
            continue

    return lattices
