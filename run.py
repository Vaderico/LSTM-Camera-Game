import argparse
import os
import sys

from utils import query_yes_no
from game import Game
from player import LSTMPlayer

# Ensures the images directory exists, and its contents is valid
def check_images_directory(directory):
    directory = directory.strip('/') + '/'

    # Ensures image directory exists
    if not os.path.exists(directory):
        sys.exit("Error: Directory doesn't exist at path: " + directory)

    # Checks for meta.json 
    if not os.path.exists(directory + 'meta.json'):
        sys.exit("Error: meta.json doesn't exist in directory: " + 
                 directory + "\nRun process_images.py to split the data.")

    return directory

# Ensures the models directory and is valid
def check_model_directory(directory):
    directory = directory.strip('/') + '/'

    # Ensures models directory exists
    if not os.path.exists(directory):
        sys.exit("Error: Directory doesn't exist at path: " + directory)

    # Ensure directory is not empty
    if len(os.listdir(directory)) == 0:
        sys.exit("Error: Directory is empty");

    # Ensure directory contains model
    model_path = os.path.join(directory, "model.pt")
    if not os.path.exists(model_path):
        sys.exit("Error: Directory %s does not contain model: " % directory)

    return model_path

# Main function
if __name__ == "__main__" :
    # Argument parsing gets image directory, training data directory, and window size
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="name of images directory to record data from")
    parser.add_argument("models", help="name of models directory to run from")
    parser.add_argument("-f", "--frame-rate", dest="frame_rate", type=int, default=30, help="frame rate of video recording")
    args = parser.parse_args()

    # Check validity of image and models data directories/files
    images_dir = check_images_directory(args.images)
    models_dir = check_model_directory(args.models)

    # Create and run program 
    player = LSTMPlayer(models_dir)
    game = Game(player, images_dir, 600, args.frame_rate)
    game.on_execute()

