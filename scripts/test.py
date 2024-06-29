#%%

from siim_ProbUNet_datamodule_masks import SIIMDataModule

data_module = SIIMDataModule(batch_size_train = 1, num_workers_train=2)
train_data_loader = data_module.train_dataloader()

#%%
from tqdm import tqdm

counter = 0
for data in tqdm(train_data_loader):
    print(data['input'].shape)
    print(data['target'].shape)
    print(data['mask'].shape)
    counter += 1
    if counter == 2:
        break

#%%

import matplotlib.pyplot as plt

plt.imshow(data['input'][0, 0, :, :].T, cmap='gray')
#%%
plt.imshow(data['target'][0, 0, :, :].T, cmap='plasma')
#%%
plt.imshow(data['mask'][0, 0, :, :].T, cmap='plasma')

#%%