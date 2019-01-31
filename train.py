import argparse
import os
import sys

from utils import query_yes_no
from train_manager import TrainManager
# from train_manager_2 import TrainManager

# Ensures the training data directory extists, and deletes its contents
def check_training_directory(directory):
    directory = directory.strip('/') + '/'

    # Ensures training data directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory

    # Checks that the training data is not empty
    if len(os.listdir(directory)) is 0:
        sys.exit("Error: Directory %s is empty" % directory)

    return directory

# Ensures the models data directory extists, and deletes its contents
def check_models_directory(directory):
    directory = directory.strip('/') + '/'

    # Ensures training data directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory

    # Checks that the training data is empty
    if len(os.listdir(directory)) > 0:
        query = "Directory: " + directory + " already contains a model, would you like to clear it and continue?"
        if not query_yes_no(query):
            sys.exit()

    return directory

# Main function
if __name__ == "__main__" :
    # Argument parsing gets image directory, training data directory, and window size
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help="name of directory to load training data from")
    parser.add_argument("models", help="name of directory to save model into")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("-s", "--split", dest="split", type=float, default=0.2, help="percentage of data for validation and test set")
    args = parser.parse_args()

    # Check validity of image and training data directories/files
    training_dir = check_training_directory(args.training)
    # Check to see if model directory exists
    models_dir = check_models_directory(args.models)

    # Create trainer, load the data, and train the model
    train_manager = TrainManager(training_dir, models_dir, args.split, args.epochs)
    train_manager.load_data()
    train_manager.train_model()

