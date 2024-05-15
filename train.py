from tensorflow.keras.models import Sequential

from process_data import concatenate_data_parts, split_data


def train_and_save_model(model: Sequential,
                         data_path: str,
                         model_save_path: str,
                         batch_size: int = 32,
                         epochs: int = 50,
                         data_pos: int = 4,
                         label_pos: int = -1) -> None:
    """
    Trains the model using data from the specified path and saves the trained model.

    Args:
    data_path (str): The file path to the dataset.
    model_save_path (str): The file path where the trained model will be saved.
    """
    # Load and split data
    data, labels = concatenate_data_parts(data_path, data_pos, label_pos)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(data, labels, data_path)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # Save the trained model
    print(f'Saving model to: {model_save_path}')
    model.save(model_save_path)