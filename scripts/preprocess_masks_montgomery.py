#%%

import os
import cv2
from glob import glob

data_root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet'
image_dir = os.path.join(data_root, 'CXR_png')
label_dir = os.path.join(data_root, 'ManualMask')
left_masks_dir = os.path.join(label_dir, 'leftMask')
right_masks_dir = os.path.join(label_dir, 'rightMask')
combined_masks_dir = os.path.join(label_dir, 'combinedMask')

if not os.path.exists(combined_masks_dir):
    os.makedirs(combined_masks_dir)

def combine_masks(left_mask_path, right_mask_path, output_path):
    left_mask = cv2.imread(left_mask_path, cv2.IMREAD_GRAYSCALE)
    right_mask = cv2.imread(right_mask_path, cv2.IMREAD_GRAYSCALE)
    combined_mask = cv2.bitwise_or(left_mask, right_mask)
    cv2.imwrite(output_path, combined_mask)

# List all left mask files
left_mask_files = glob(os.path.join(left_masks_dir, '*.png'))

# Process each file
for left_mask_path in left_mask_files:
    base_filename = os.path.basename(left_mask_path)
    right_mask_path = os.path.join(right_masks_dir, base_filename)
    output_path = os.path.join(combined_masks_dir, base_filename)
    
    combine_masks(left_mask_path, right_mask_path, output_path)


#%%





#%%






#%%