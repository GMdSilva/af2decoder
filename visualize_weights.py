import random
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model

from layers.attentionLayer import AttentionLayer

matplotlib.use('TkAgg')
random.seed(42)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def analyze_attention_weights(model: Sequential,
                              x_test: np.ndarray,
                              cutoff: int = 99) -> dict:
    """
    Visualizes the attention weights from a specified layer in a neural network model.

    Args:
    model (Model): The trained Keras model from which to extract attention weights.
    x_test (np.ndarray): The test dataset used to evaluate the model.
    cutoff (int): The percentile above which to consider attention weights as significant.

    Returns:
    dict: A dictionary containing:
        - 'transposed_weights': The raw transposed attention weights.
        - 'mean_attention_weights_per_pos': Sum of absolute attention weights across all positions.
        - 'filtered_weights': Attention weights filtered by the given cutoff percentile.
    """
    # Extract the attention layer's output from the model.
    attention_layer_output = model.get_layer('attention_layer').output
    attention_model = Model(inputs=model.input, outputs=attention_layer_output)

    # Predict attention weights using the test data.
    attention_weights = attention_model.predict(x_test)

    # Calculate the mean of attention weights across samples.
    mean_attention_weights = np.mean(attention_weights, axis=0)
    absolute_attention_weights = np.abs(mean_attention_weights)

    # Sum absolute attention weights across all amino acid positions.
    mean_attention_weights_per_position = np.sum(absolute_attention_weights, axis=1)

    # Transpose weights for easier manipulation.
    transposed_weights = mean_attention_weights.T

    # Determine the cutoff value for significant weights.
    positive_weights = transposed_weights[transposed_weights > 0]
    cutoff_value = np.percentile(positive_weights, cutoff)

    # Filter weights below the cutoff, setting them to NaN.
    filtered_weights = np.where(transposed_weights > cutoff_value, transposed_weights, np.nan)

    # Store and return the results in a dictionary.
    visualization_results = {
        'transposed_weights': transposed_weights,
        'mean_attention_weights_per_position': mean_attention_weights_per_position,
        'filtered_weights': filtered_weights
    }

    return visualization_results


def analyze_visualization(model_path: str,
                          data_path: str,
                          trial_name: str = 'wt',
                          plot_results=True,
                          save_top_positions=True) -> None:
    """
    Analyzes and visualizes attention weights from a trained model for specific data.

    Args:
    model_path (str): Path to the saved Keras model.
    data_path (str): Path to the dataset.
    aa_len (int): Length of amino acids to consider for visualization.
    test (str): Descriptor for the test condition to name output files.

    Returns:
    None: The function outputs visualizations and saves analysis results to a file.
    """
    # Load the model with custom attention layer.
    model = load_model(f'{data_path}/results/{model_path}',
                       custom_objects={'AttentionLayer': AttentionLayer})

    # Load and prepare data.
    x_test = load_pickle(f'{data_path}/test_sets/test_set_data.h5')

    # Visualize attention weights.
    visualization_results = analyze_attention_weights(model, x_test)

    # Define amino acid labels.
    amino_acid_index = {
        "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3,
        "CYS": 4, "GLN": 5, "GLU": 6, "GLY": 7,
        "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11,
        "MET": 12, "PHE": 13, "PRO": 14, "SER": 15,
        "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19, "UNK": 20, "GAP": 21, "TOK": 22
    }
    labels = [k for k, v in sorted(amino_acid_index.items(), key=lambda item: item[1])]
    aa_len = range(x_test.shape[1])

    if plot_results:

        # Setup plot for heatmap of filtered weights.
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(visualization_results['filtered_weights'],
                        cmap='viridis',
                        aspect='auto',
                        interpolation='nearest')

        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, rotation=0)
        ax.set_ylabel('Amino Acid')
        ax.set_xlabel('Position Index')
        plt.savefig(f'{data_path}/results/att_weights_{trial_name}.png')
        plt.show()

        # Plot aggregated mean attention weights.
        plt.figure(figsize=(15, 5))
        plt.bar(aa_len, visualization_results['mean_attention_weights_per_position'])
        plt.title('Aggregated Mean Attention Weights Per Position')
        plt.xlabel('Residue Position')
        plt.ylabel('Aggregated Attention Weight')
        plt.savefig(f'{data_path}/results/aggregated_att_weights_{trial_name}.png')
        plt.show()

    if save_top_positions:

        # Save significant weights and corresponding labels to a file.
        rows, cols = np.where(visualization_results['filtered_weights'] >= 0)
        values = visualization_results['filtered_weights'][rows, cols]
        amino_acids = [labels[row] for row in rows]
        positions = cols

        with open(f'{data_path}/results/top_attention_weights_{trial_name}.txt', 'w') as file:
            file.write("Amino Acid,Position,Value\n")
            for aa, pos, val in zip(amino_acids, positions, values):
                file.write(f"{aa},{pos},{val}\n")
