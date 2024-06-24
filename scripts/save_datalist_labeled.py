#%%

import os, json

data_root: str = '/data2/open_dataset/chest_xray/SIIM_TRAIN_TEST/Pneumothorax'
json_file: str = 'datalist.json'
json_path = os.path.join(data_root, json_file)

with open(json_path) as f:
    data_list = json.load(f)

#%%


new_data_list = {}
for key, entries in data_list.items():
    new_entries = [entry for entry in entries if 'empty' not in entry['label']]
    new_data_list[key] = new_entries

personal_root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data'
new_json_path = os.path.join(personal_root, 'datalist_labeled.json')
with open(new_json_path, 'w') as f:
    json.dump(new_data_list, f)


#%%

import os, json
personal_root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data'
json_path = os.path.join(personal_root, 'datalist_labeled.json')

with open(json_path, 'r') as f:
    data_list = json.load(f)

data_list

#%%

for key, entries in data_list.items():
    print(key, len(entries))

#%%