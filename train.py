import argparse
import os

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

# Main function
if __name__ == "__main__" :
    # Argument parsing gets image directory, training data directory, and window size
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help="name of directory to load training data from")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=10, help="size of batches")
    parser.add_argument("-v", "--validation-split", dest="val_split", type=float, default=0.2, help="percentage of data for validation set")
    args = parser.parse_args()

    print(args.frame_rate)

    # Check validity of image and training data directories/files
    training_dir = check_training_directory(args.training)

    train_manager = TrainManager(training_dir, arg.val_split, args.epochs, args.batch_size)
    train_manager.convert_data()
    train_manager.train_model()
    train_manager.save_model()

