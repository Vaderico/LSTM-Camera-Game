import argparse
import json
import glob
import os
import sys
import random
from utils import query_yes_no

# Check/clean directory, split train/test, and create meta file
def process_images(directory, split):
    base_path = directory.strip('/') + '/'

    # Checks for existance of directory
    if not os.path.exists(directory):
        sys.exit("Error: can't find directory: " + base_path)

    # Checks for existance of meta file, and delete when prompted
    meta_filename = "meta.json"
    meta_path = base_path + meta_filename
    if os.path.exists(meta_path):
        if query_yes_no("Images already processed. Would you like to overwrite it?"):
            os.remove(meta_path)
        else:
            sys.exit()
    
    meta_data = rename_files(base_path, split)

    # Store file splits into json
    with open(meta_path, 'w') as outfile:  
        json.dump(meta_data, outfile, indent=2)

# Shuffle files and split into train/test
def rename_files(base_path, split):
    clean_directory(base_path)

    # Append ~ to all filenames
    for filename in os.listdir(base_path):
        os.rename(base_path + filename, base_path + '~' + filename)

    # Shuffle files
    filenames = os.listdir(base_path)
    random.shuffle(filenames)

    # Calculate number of training and test images
    n_files = len(filenames)
    n_test = int(n_files * split)
    n_train = n_files - n_test
    
    # Extract extensions from all filenames
    exts = [os.path.splitext(name)[1] for name in filenames]

    # Create names for new train and test files
    new_test_names = ["test_%d%s" % (x+1, exts[x]) for x in range(n_test)]
    new_train_names = ["train_%d%s" % (x+1, exts[x+n_test]) for x in range(n_train)]
    new_names = new_test_names + new_train_names

    # Rename files to train and test names
    for old, new in zip(filenames, new_names):
        os.rename(base_path + old, base_path + new)

    # Return all train and test names to be stored as json
    return { "training_images": new_train_names, "test_images": new_test_names }

# Remove files with non-image extensions
def clean_directory(base_path):
    # Allowed extensions
    image_extensions = [".jpg", ".jpeg", ".png"]
    filenames = os.listdir(base_path)

    # Delete all files without allowed extension
    for filename in filenames:
        f, ext = os.path.splitext(filename)
        if not ext in image_extensions:
            os.remove(base_path + filename)

# Main function
if __name__ == "__main__" :
    # Argument parsing to take image folder and split threshold
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="name of folder of images to record data from")
    parser.add_argument("-s", "--split", dest="split", type=float, default=0.2, help="percentage of data to split into test")
    args = parser.parse_args()

    # Parse image folder and split filenames
    process_images(args.images, args.split)

