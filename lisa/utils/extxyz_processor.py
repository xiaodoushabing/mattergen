"""
This module provides functions for visualizing atomic structures using py3Dmol
and processing data from .extxyz files.  It includes functions to render
interactive 3D visualizations, parse structure data, and select specific
structures for visualization.
"""
import pandas as pd
from .visualisation import visualise_structure

# %% convert generated .extxyz file into something readable
def conv_results(structures_path, preview = False):
    """Parses and processes structure data from an .extxyz file.

    Args:
        structures_path (str): Path to the .extxyz file.
        preview (bool, optional): If True, prints the parsed data to the console. Defaults to False.

    Returns:
        tuple: A tuple containing two dictionaries:
            - long_results: A dictionary containing detailed information about each structure,
              including atom count, lattice information, raw data lines, and a list of atom types
              and their counts.  Keys are integer indices.
            - short_results_df: A Pandas DataFrame containing summary information about each structure,
              including atom count and a list of atom types and their counts.  Indexed by structure
              number.
    """
    with open(structures_path, "r") as f:
        lines = f.read().strip().split("\n")

    long_results = {}
    i = 0
    idx = 1

    # process the lines and lattices
    while i < len(lines):
        atom_count = int(lines[i].strip())
        lattice = lines[i + 1].strip()
            
        # Get the data corresponding to this lattice
        datas = lines[i + 2:i + 2 + atom_count]
        
        atom_list = {}
        for data in datas:
            atom = data.split(" ")[0]
            atom_list[atom] = atom_list.get(atom, 0) + 1

        # Store the data in the lattices dictionary
        long_results[idx] = {
            'atom_count': atom_count,
            'lattice': lattice,
            'data': datas,
            'atom_list': atom_list}
        
        # Move to the next section
        i += 2 + atom_count
        # increment index
        idx += 1

    short_results = {idx: {key: long_results[idx][key] for key in ['atom_count', 'atom_list']} for idx in long_results}
    short_results_df = pd.DataFrame.from_dict(short_results).T
    
    if preview:
        print("="*50,"Long Results","="*50)
        for idx, info in long_results.items():
            print(f"Index: {idx}")
            print(f"Atom count: {info['atom_count']}")
            print(f"Lattice: {info['lattice']}")
            print(f"Data: {info['data']}")
            print(f"Atom List: {info['atom_list']}")
            print("=" * 50)
            if idx == 2:
                break
        
        print("\n\n","="*50,"Short Results","="*50)
        for idx, info in short_results.items():
            print(f"Index: {idx}")
            print(f"Atom count: {info['atom_count']}")
            print(f"Atom List: {info['atom_list']}")
            print("=" * 50)
            if idx == 2:
                break

        print("\n\n","="*50,"Short Results DF","="*50)
        print(short_results_df.head())
        
    return long_results, short_results_df

# %%
def pick_results(idx, long_results, structure_path, repeat_unit=3, visualise = True):
    """Selects a specific structure from the parsed data and writes it to a file,
    optionally visualizing it.

    Args:
        idx (int): The index of the structure to select.
        long_results (dict): The dictionary of parsed structure data (from conv_results).
        structures_path (str): Path to the output .extxyz file.
        visualise (bool, optional): If True, visualizes the selected structure. Defaults to True.

    Raises:
        KeyError: If the provided index `idx` is not found in `long_results`.
    """
    if idx not in long_results:
        raise KeyError(f"Index {idx} not found in long_results.")
    with open(structure_path, "w") as f:
        f.write(f"{long_results[idx]['atom_count']}\n")
        f.write(f"{long_results[idx]['lattice']}\n")
        f.writelines("\n".join(long_results[idx]['data']) + "\n")
    if visualise:
        _ = visualise_structure(structure_path,repeat_unit=repeat_unit)
