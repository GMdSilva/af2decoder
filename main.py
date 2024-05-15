from model import ModelBuilder
from params import CONS, input_dict
from train import train_and_save_model
from visualize_weights import analyze_visualization

msa_model = ModelBuilder(input_dict['seq_len'],
                         CONS['dense_layers'],
                         CONS['dropout_rate']).create_model()

train_and_save_model(msa_model,
                     input_dict['data_path'],
                     input_dict['save_path'],
                     CONS['batch_size'],
                     CONS['epochs'],
                     CONS['data_pos'],
                     CONS['label_pos'])

analyze_visualization(input_dict['save_path'],
                      input_dict['data_path'],
                      input_dict['trial'])
