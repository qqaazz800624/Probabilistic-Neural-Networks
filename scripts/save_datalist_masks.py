#%%
import os, json

personal_root: str = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data'
json_file: str = 'datalist_labeled.json'
json_path = os.path.join(personal_root, json_file)

with open(json_path) as f:
    data_list = json.load(f)

data_list

#%%
folds = ['training', 'validation', 'testing']
for fold in folds:
    for item in data_list[fold]:
        image_path = item['image']
        base_name = os.path.basename(image_path)
        uncertaunty_mask_path = f"data/Masks/{base_name}"
        item['mask'] = uncertaunty_mask_path

#%%

with open(os.path.join(personal_root, 'datalist_masks.json'), 'w') as f:
    json.dump(data_list, f)


#%%


