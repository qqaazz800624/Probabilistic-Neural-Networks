#%%

import torch
from siim_datamodule import SIIMDataModule
from tqdm import tqdm
from uncertainty_heatmap import UncertaintyHeatmap

#%%

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data_module = SIIMDataModule(batch_size_test=16, num_workers_test=2)
test_loader = data_module.test_dataloader()

counter = 0
with torch.no_grad():
    for data in tqdm(test_loader):
        img, label = data['input'].to(device), data['target']
        mask_uncertainty = UncertaintyHeatmap().__call__(img, label)
        counter += 1
        if counter == 1:
            break


#%%

mask_uncertainty

#%%




#%%






#%%






#%%