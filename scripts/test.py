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
    if counter == 1:
        break

#%%







#%%