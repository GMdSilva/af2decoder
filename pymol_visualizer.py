import pymol
from pymol import cmd
from params import input_dict

def visualize_on_pymol(reference_path: str,
                       attention_weight_results_path: str,
                       save_path: str,
                       show_sticks=True,
                       palette: str = 'yellow_orange_red'):

    # Initialize PyMOL
    pymol.finish_launching()

    # Load your structure file
    cmd.load(reference_path)

    # Function to read positions and values from file
    def read_positions(file_name):
        positions = []
        values = []
        with open(file_name, 'r') as file:
            next(file)  # Skip header line
            for line in file:
                parts = line.strip().split(',')
                pos = int(parts[1]) + 1  # Position, incremented by 1
                value = float(parts[2])  # B-factor value
                positions.append(pos)
                values.append(value)
        return positions, values

    # Read positions and b-factor values from the file
    positions, values = read_positions(attention_weight_results_path)

    # Rescale b-factor values from 0 to 1
    min_value = min(values)
    max_value = max(values)

    # Apply the scaled b-factors and change representation
    for pos, value in zip(positions, values):
        scaled_value = (value - min_value) / (max_value - min_value)  # Scale b-factor
        selection = f"resi {pos}"
        cmd.alter(selection, f'b={scaled_value}')  # Set b-factor to scaled value
        if show_sticks:
            cmd.show("sticks", selection)

    # Apply a color spectrum based on the new b-factor scaling
    cmd.spectrum("b", palette, " or ".join([f"resi {pos}" for pos in positions]))

    # Zoom in on the modified residues
    cmd.zoom(" or ".join([f"resi {pos}" for pos in positions]))

    # Save the session if needed
    cmd.save(save_path)

visualize_on_pymol(f"{input_dict['data_path']}/reference.pdb",
                   f"{input_dict['data_path']}/results/top_attention_weights_{input_dict['trial']}.txt",
                   f"{input_dict['data_path']}/results/pymol_visualization_{input_dict['trial']}.se")