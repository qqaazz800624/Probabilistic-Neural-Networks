# Import required libraries
import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
 
"""
We've downloaded the "train" folder. The original train set images lie inside
the "original" folder.
"""
# Define paths for training dataset and image directory
TRAIN_CSV = "/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/UW/train.csv"
ORIG_IMG_DIR = os.path.join("/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/UW", "train")
CASE_FOLDERS = os.listdir(ORIG_IMG_DIR)
 
# Define paths for training and validation image and mask directories
ROOT_DATASET_DIR = "/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/UW"
ROOT_TRAIN_IMG_DIR = os.path.join(ROOT_DATASET_DIR, "train", "images")
ROOT_TRAIN_MSK_DIR = os.path.join(ROOT_DATASET_DIR, "train", "masks")
ROOT_VALID_IMG_DIR = os.path.join(ROOT_DATASET_DIR, "valid", "images")
ROOT_VALID_MSK_DIR = os.path.join(ROOT_DATASET_DIR, "valid", "masks")
 
# Create directories if not already present
os.makedirs(ROOT_TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(ROOT_TRAIN_MSK_DIR, exist_ok=True)
os.makedirs(ROOT_VALID_IMG_DIR, exist_ok=True)
os.makedirs(ROOT_VALID_MSK_DIR, exist_ok=True)
 
# Define regular expressions to extract case, date, slice number, and image shape from file paths
GET_CASE_AND_DATE = re.compile(r"case[0-9]{1,3}_day[0-9]{1,3}")
GET_SLICE_NUM = re.compile(r"slice_[0-9]{1,4}")
IMG_SHAPE = re.compile(r"_[0-9]{1,3}_[0-9]{1,3}_")
 
# Load the main dataframe from csv file and drop rows with null values
MAIN_DF = pd.read_csv(TRAIN_CSV).dropna(axis=0)
only_IDS = MAIN_DF["id"].to_numpy()
 
# Define classes for image segmentation
CLASSES = ["large_bowel", "small_bowel", "stomach"]
 
# Create a mapping of class ID to RGB value
color2id = {
    (0, 0, 0): 0,  # background pixel
    (0, 0, 255): 1,  # Blue - Stomach
    (0, 255, 0): 2,  # Green - Small bowel
    (255, 0, 0): 3,  # Red - Large bowel
}
 
# Reverse map from id to color
id2color = {v: k for k, v in color2id.items()}
 
 
# Function to get all relevant image files in a given directory
def get_folder_files(folder_path):
    all_relevant_imgs_in_case = []
    img_ids = []
 
    for dir, _, files in os.walk(folder_path):
        if not len(files):
            continue
 
        for file_name in files:
            src_file_path = os.path.join(dir, file_name)
 
            case_day = GET_CASE_AND_DATE.search(src_file_path).group()
            slice_id = GET_SLICE_NUM.search(src_file_path).group()
            image_id = case_day + "_" + slice_id
 
            if image_id in only_IDS:
                all_relevant_imgs_in_case.append(src_file_path)
                img_ids.append(image_id)
 
    return all_relevant_imgs_in_case, img_ids
 
 
# Function to decode Run-Length Encoding (RLE) into an image mask
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
 
    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction
 
 
# Function to load and convert image from a uint16 to uint8 datatype.
def load_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    return img
 
 
# Function to convert RGB image to one-hot encoded grayscale image based on color map.
def rgb_to_onehot_to_gray(rgb_arr, color_map=id2color):
    num_classes = len(color_map)
    shape = rgb_arr.shape[:2] + (num_classes,)
    arr = np.zeros(shape, dtype=np.float32)
 
    for i, cls in enumerate(color_map):
        arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == color_map[i], axis=1).reshape(shape[:2])
 
    return arr.argmax(-1)
 
 
# Function to create and write image-mask pair for each file path in given directories.
def create_and_write_img_msk(file_paths, file_ids, save_img_dir, save_msk_dir, desc=None):
    for file_path, file_id in tqdm(zip(file_paths, file_ids), ascii=True, total=len(file_ids), desc=desc, leave=True):
        image = load_img(file_path)
 
        IMG_DF = MAIN_DF[MAIN_DF["id"] == file_id]
 
        img_shape_H_W = list(map(int, IMG_SHAPE.search(file_path).group()[1:-1].split("_")))[::-1]
        mask_image = np.zeros(img_shape_H_W + [len(CLASSES)], dtype=np.uint8)
 
        for i, class_label in enumerate(CLASSES):
            class_row = IMG_DF[IMG_DF["class"] == class_label]
 
            if len(class_row):
                rle = class_row.segmentation.squeeze()
                mask_image[..., i] = rle_decode(rle, img_shape_H_W) * 255
 
        mask_image = rgb_to_onehot_to_gray(mask_image, color_map=id2color)
 
        FILE_CASE_AND_DATE = GET_CASE_AND_DATE.search(file_path).group()
        FILE_NAME = os.path.split(file_path)[-1]
 
        new_name = FILE_CASE_AND_DATE + "_" + FILE_NAME
 
        dst_img_path = os.path.join(save_img_dir, new_name)
        dst_msk_path = os.path.join(save_msk_dir, new_name)
 
        cv2.imwrite(dst_img_path, image)
        cv2.imwrite(dst_msk_path, mask_image)
 
    return
 
 
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
 
    # Main script execution: for each folder, split the data into training and validation sets, and create/write image-mask pairs.
    for folder in CASE_FOLDERS:
        all_relevant_imgs_in_case, img_ids = get_folder_files(folder_path=os.path.join(ORIG_IMG_DIR, folder))
        train_files, valid_files, train_img_ids, valid_img_ids = train_test_split(all_relevant_imgs_in_case, img_ids, train_size=0.8, shuffle=True)
        create_and_write_img_msk(train_files, train_img_ids, ROOT_TRAIN_IMG_DIR, ROOT_TRAIN_MSK_DIR, desc=f"Train :: {folder}")
        create_and_write_img_msk(valid_files, valid_img_ids, ROOT_VALID_IMG_DIR, ROOT_VALID_MSK_DIR, desc=f"Valid :: {folder}")
        print()