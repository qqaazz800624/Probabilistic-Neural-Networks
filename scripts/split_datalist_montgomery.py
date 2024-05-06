#%%


import os
import json
from glob import glob
from sklearn.model_selection import KFold

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
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        mask_path = os.path.join(mask_dir, filename)
        
        # Check if corresponding mask exists
        if os.path.exists(mask_path):
            data_list.append({
                'filename': filename,
                'image': os.path.join('CXR_png', filename),
                'taget': os.path.join('ManualMask/combinedMask', filename)
            })

    five_fold_datalist = {}
    for k, (_, fold) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(data_list)):
        five_fold_datalist[f"fold_{k}"] = [data_list[i] for i in fold]
    
    # Write to JSON file
    # with open(os.path.join(data_root, 'datalist_montgomery.json'), 'w') as outfile:
    #     json.dump(data_list, outfile, indent=4)

    with open(os.path.join(data_root, 'datalist_fold_montgomery.json'), 'w') as outfile:
        json.dump(five_fold_datalist, outfile, indent=4)

# Call the function
#create_data_list(image_dir, combined_masks_dir)

if __name__ == '__main__':
    main()

#%%






#%%






#%%