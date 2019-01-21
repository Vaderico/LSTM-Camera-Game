import argparse
import os
import sys
import csv

from utils import query_yes_no
from train_manager import TrainManager

def step_vel_file(step):
    file = []
    small = True
    for i in range(100):
        if i % step is 0:
            small = not small
        if (small):
            file.append([6,0])
        else:
            file.append([0,6])
    return file

# Main function
if __name__ == "__main__" :
    # Argument parsing gets image directory, training data directory, and window size
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="name of directory to edit")
    parser.add_argument("-s", "--step-size", dest="step", type=int, help="Step size")
    args = parser.parse_args()

    paths = os.listdir(args.directory)
    steps = step_vel_file(args.step)

    # Write recorded velocities to file
    for folder in paths:
        path_name = os.path.join(args.directory, folder, "velocities.csv")
        with open(path_name, 'w') as csv_file:  
            csv.writer(csv_file, delimiter=',').writerows(steps)
    

