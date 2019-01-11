import argparse
import os
import sys
import shutil

from utils import query_yes_no
from game import RecordGame, Game
from human_player import HumanPlayer
from lstm_player import LSTMPlayer

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

# Ensures the training data directory extists, and deletes its contents
def check_training_directory(directory):
    directory = directory.strip('/') + '/'

    # Ensures training data directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory

    # Deletes contents of directory if not empty
    if len(os.listdir(directory)) > 0:
        query = "Directory: " + directory + " already contains training data, would you like to add to it?"
        if not query_yes_no(query):
            sys.exit()

    return directory

# Main function
if __name__ == "__main__" :
    # Argument parsing gets image directory, training data directory, and window size
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="name of images directory to record data from")
    parser.add_argument("training", help="name of directory to save recorded training data to")
    parser.add_argument("-r", "--resolution", dest="res", type=int, default=600, help="size of window displaying the game")
    parser.add_argument("-m", "--max-velocity", dest="max_vel", type=float, default=7, help="maximum velocity of camera movement")
    parser.add_argument("-a", "--acceleration", dest="accel", type=float, default=0.6, help="acceleration of camera movement")
    parser.add_argument("-d", "--deceleration", dest="decel", type=float, default=0.8, help="deceleration of camera movement")
    parser.add_argument("-f", "--frame-rate", dest="frame_rate", type=int, default=30, help="frame rate of video recording")
    parser.add_argument("-s", "--manual-stop", dest="stop", action="store_true", help="allows recordings to be stopped manually with spacebar")
    parser.add_argument("-x", "--max-frames", dest="max_frames", type=int, default=1000, help="automatically stops recording after given number of frames")
    args = parser.parse_args()

    print(args.frame_rate)

    # Check validity of image and training data directories/files
    images_dir = check_images_directory(args.images)
    training_dir = check_training_directory(args.training)

    # Create and run progra to record training data
    player = HumanPlayer(args.max_vel, args.accel, args.decel)
    # player = LSTMPlayer()

    game = RecordGame(player, images_dir, training_dir, args.res, args.frame_rate, args.stop, args.max_frames)
    # game = Game(player, images_dir, args.res, args.frame_rate)

    game.on_execute()
