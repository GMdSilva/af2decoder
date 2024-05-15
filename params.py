epochs = 100
batch_size = 8
dense_layers = 200
dropout_rate = 0.1
data_pos = 4
label_pos = -1

input_tuple_codex = {
	0: 'seed',
	1: 'af2_model',
	2: 'prev_msa_first_row',
	3: 'prev_dgram',
	4: 'msa_logit',
	-1: 'labels'
}

trial = 'abl1_gr_vs_i2_variants'
data_path = 'abl1_datasets'
seq_len = 260
save_path = f'{trial}_ep_{epochs}_ba_{batch_size}_dl_{dense_layers}_dr_{dropout_rate}_{input_tuple_codex[data_pos]}.h5'

CONS = {
	'epochs': epochs,
	'batch_size': batch_size,
	'dense_layers': dense_layers,
	'dropout_rate': dropout_rate,
	'data_pos': data_pos,  # for indexing from the input tuple
	'label_pos': label_pos  # for indexing from the input tuple
}

input_dict = {
	'trial': trial,
	'data_path': data_path,
	'seq_len': seq_len,
	'save_path': save_path
}
