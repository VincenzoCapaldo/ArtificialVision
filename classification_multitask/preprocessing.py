# this script resize the image in the size specified by the DatasetConfig class (see configuration_parameters.py)

import os
from PIL import Image, ImageOps
from configuration_parameters import DatasetConfig
from tqdm import tqdm
import shutil

# config has all the preprocessing configuration parameters
config = DatasetConfig
# path of the training and test sets
TRAIN_IMG_PATH = '../../Dataset/training_set/training_set'
TRAIN_LABEL_PATH = '../../Dataset/training_set.txt'

TEST_IMG_PATH = '../../Dataset/validation_set/validation_set'
TEST_LABEL_PATH = '../../Dataset/validation_set.txt'
VALIDATION_LABEL_PATH = '../../PreprocessedDataset/val_split.txt'
# new dataset location
OUTPUT_PATH = "../../PreprocessedDataset/"
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "training_set/")
TEST_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "test_set/")
VALIDATION_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "validation_set/")

# create directories if needed
os.makedirs(TRAIN_OUTPUT_PATH, exist_ok=True)
os.makedirs(TEST_OUTPUT_PATH, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_PATH, exist_ok=True)


def resize(img, target_size=config.IMAGE_SIZE, padding=False):
    if not padding:
        return img.resize(target_size)
    else:
        img.thumbnail(target_size)  # Scale while maintaining the ratio
        delta_width = target_size[0] - img.size[0]
        delta_height = target_size[1] - img.size[1]
        padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)
        return ImageOps.expand(img, padding, fill=(0, 0, 0))  # add black pixels


def preprocess_and_save_images(input_path, output_path, target_size=config.IMAGE_SIZE, padding=False):

    total_files = sum([len(files) for _, _, files in os.walk(input_path) if files])
    # add a progress bar in the console
    with tqdm(total=total_files, desc=f"Processing {input_path}", unit="file") as pbar:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter images only
                    input_file_path = os.path.join(root, file)

                    relative_path = os.path.relpath(root, input_path)
                    output_dir = os.path.join(output_path, relative_path)
                    os.makedirs(output_dir, exist_ok=True)

                    output_file_path = os.path.join(output_dir, file)

                    # process the image
                    try:
                        with Image.open(input_file_path) as img:
                            img = img.convert("RGB")  # convert in RGB
                            img = resize(img, target_size, padding)
                            img.save(output_file_path)
                    except Exception as e:
                        print(f"Fail while processing {input_file_path}: {e}")
                    pbar.update(1)

#validation_file =  '../../PreprocessedDataset/val_split.txt'
# input path = "../../PreprocessedDataset/training_set"
#validation output path = "../../PreprocessedDataset/validation_set"
def move_validation_images(validation_file, training_path, validation_output_path):
    """
    Reads file names from a validation split file and moves corresponding images from the training folder
    to the validation folder.

    :param validation_file: Path to the validation split file (CSV format, first column contains file names).
    :param training_path: Path to the training images folder.
    :param validation_output_path: Path to the validation images folder.
    """
    # Read the validation file
    with open(validation_file, 'r') as f:
        image_names = [line.strip().split(',')[0] for line in f.readlines()]

    os.makedirs(validation_output_path, exist_ok=True)

    # Move each image
    for image_name in tqdm(image_names, desc="Moving validation images", unit="file"):
        input_file_path = os.path.join(training_path, image_name)
        output_file_path = os.path.join(validation_output_path, image_name)

        if os.path.exists(input_file_path):
            try:
                shutil.move(input_file_path, output_file_path)
            except Exception as e:
                print(f"Failed to move {input_file_path}: {e}")
        else:
            print(f"Image {input_file_path} does not exist.")

# PADDING = False
# print("Preprocessing of the training_set...")
# preprocess_and_save_images(TRAIN_IMG_PATH, TRAIN_OUTPUT_PATH, config.IMAGE_SIZE, PADDING)
#
# print("Preprocessing of the test_set (ex validation_set)...")
# preprocess_and_save_images(TEST_IMG_PATH, TEST_OUTPUT_PATH, config.IMAGE_SIZE, PADDING)

# print("Moving validation images...")
# move_validation_images(VALIDATION_LABEL_PATH, TRAIN_OUTPUT_PATH, VALIDATION_OUTPUT_PATH)




# print(f"Preprocessed dataset stored in: {OUTPUT_PATH}")