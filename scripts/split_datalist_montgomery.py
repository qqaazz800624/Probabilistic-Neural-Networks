#%%

import os
import json
from glob import glob
from sklearn.model_selection import KFold, StratifiedKFold

data_root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet'
image_dir = os.path.join(data_root, 'CXR_png')
label_dir = os.path.join(data_root, 'ManualMask')
combined_masks_dir = os.path.join(label_dir, 'combinedMask')


def main():
    image_dir = os.path.join(data_root, 'CXR_png')
    mask_dir = os.path.join(data_root, 'ManualMask', 'combinedMask')
    # Get list of all image files
    image_files = glob(os.path.join(image_dir, '*.png'))

    # Prepare the data list
    data_list = []
    tb_status_list = []
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        mask_path = os.path.join(mask_dir, filename)
        
        # Extract TB status from filename
        tb_status = filename.split('_')[-1].split('.')[0]  # This will be '0' or '1'
        tb_status = int(tb_status)  # Convert to integer for easier handling in Python

        # Check if corresponding mask exists
        if os.path.exists(mask_path):
            data_list.append({
                'filename': filename,
                'image': os.path.join('CXR_png', filename),
                'target': os.path.join('ManualMask/combinedMask', filename),
                'tb_status': tb_status
            })
            tb_status_list.append(tb_status)  # Add tb_status to the list for stratification

    # Perform stratified K-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    five_fold_datalist = {}

    for fold_index, (train_index, test_index) in enumerate(skf.split(data_list, tb_status_list)):
        five_fold_datalist[f"fold_{fold_index}"] = [data_list[i] for i in test_index]

    # for k, (_, fold) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(data_list)):
    #     five_fold_datalist[f"fold_{k}"] = [data_list[i] for i in fold]


    with open(os.path.join(data_root, 'datalist_fold_montgomery.json'), 'w') as outfile:
        json.dump(five_fold_datalist, outfile, indent=4)

# Call the function
#create_data_list(image_dir, combined_masks_dir)

if __name__ == '__main__':
    main()

#%%

# import json
# json_path = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet/datalist_fold_montgomery.json'

# with open(json_path) as f:
#     data_list = json.load(f)

# len(data_list['fold_4'])

#%%




#%%