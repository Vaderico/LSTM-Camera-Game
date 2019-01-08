from data_converter import DataConverter

class TrainManager():
    def __init__(training_dir, val_split, epochs, batch_size):
        self._training_dir = training_dir
        self._val_split = val_split
        self._epochs = epochs
        self._batch_size = batch_size

    def convert_data(self):
        pass

    def train_model(self):
        pass
