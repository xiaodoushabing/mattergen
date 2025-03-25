
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

            for idx, data in enumerate(datas, start=1):
                split_data = data.split()
                atom = split_data[0]
                atom_list[atom] = atom_list.get(atom, 0) + 1
                atoms_data[idx] = {atom: list(map(float, split_data[1:]))}
            
            lattices.append({
                "no_of_atoms": atom_count,
                "cell_params": cell_params,
                "pbc": pbc,
                "atoms_list": atom_list,
                "atoms": atoms_data,
            })
            i += 2 + atom_count
        
        except IndexError as e:
            print(f"ERROR: IndexError encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            continue

        except ValueError as e:
            print(f"ERROR: ValueError encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            continue

        except Exception as e:
            print(f"ERROR: Unexpected error encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            continue

    return lattices
