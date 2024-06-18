#%%

from segmentation_models_pytorch import Unet
from functools import partial
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from prob_unet_first import ProbUNet_First
from prob_unet_second import ProbUNet_Second
from prob_unet_third import ProbUNet_Third
from prob_unet_fourth import ProbUNet_Fourth
import os

from siim_datamodule import SIIMDataModule
from siim_dataset import SIIMDataset
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import json
from tqdm import tqdm


# Hyperparameters
# ============ Training setting ============= #

max_epochs = 32
model_name = 'Unet'  # Valid model_name: ['Unet', 'DeepLabV3Plus']
latent_dim = 6
beta = 10
batch_size_train = 16
batch_size_val = 16
loss_fn = 'BCEWithLogitsLoss'

# ============ Inference setting ============= #
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
num_samples = 100

root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'

unet = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(model_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    model_weight[k_new] = model_weight.pop(k)

unet.load_state_dict(model_weight)

version_prev = None

# =========================================== #

ProbUnet_First = ProbUNet_First(
    model=unet,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_34' # version_28
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First.load_state_dict(model_weight)
ProbUnet_First.eval()
ProbUnet_First.requires_grad_(False)

unet_v2 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(unet_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    unet_weight[k_new] = unet_weight.pop(k)
unet_v2.load_state_dict(unet_weight)

ProbUnet_First_v2 = ProbUNet_First(
    model=unet_v2,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_34'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v2.load_state_dict(model_weight)

model_first = ProbUnet_First_v2.model
prior_first = ProbUnet_First_v2.prior
posterior_first = ProbUnet_First_v2.posterior
fcomb_first = ProbUnet_First_v2.fcomb

ProbUnet_Second = ProbUNet_Second(
    model=model_first,
    prior_first=prior_first,
    posterior_first=posterior_first,
    fcomb_first=fcomb_first,
    prob_unet_first=ProbUnet_First,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_43'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Second.load_state_dict(model_weight)
ProbUnet_Second.eval()
ProbUnet_Second.requires_grad_(False)

unet_v3 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(unet_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    unet_weight[k_new] = unet_weight.pop(k)
unet_v3.load_state_dict(unet_weight)

ProbUnet_First_v3 = ProbUNet_First(
    model=unet_v3,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_34'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v3.load_state_dict(model_weight)
ProbUnet_First_v3.eval()
ProbUnet_First_v3.requires_grad_(False)

unet_v4 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(unet_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    unet_weight[k_new] = unet_weight.pop(k)
unet_v4.load_state_dict(unet_weight)

ProbUnet_First_v4 = ProbUNet_First(
    model=unet_v4,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_34'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v4.load_state_dict(model_weight)

model_first_v4 = ProbUnet_First_v4.model
prior_first_v4 = ProbUnet_First_v4.prior
posterior_first_v4 = ProbUnet_First_v4.posterior
fcomb_first_v4 = ProbUnet_First_v4.fcomb

ProbUnet_Second_v2 = ProbUNet_Second(
    model=model_first_v4,
    prior_first=prior_first_v4,
    posterior_first=posterior_first_v4,
    fcomb_first=fcomb_first_v4,
    prob_unet_first=ProbUnet_First_v3,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_43'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Second_v2.load_state_dict(model_weight)

model_second = ProbUnet_Second_v2.model
prior_second = ProbUnet_Second_v2.prior
posterior_second = ProbUnet_Second_v2.posterior
fcomb_second = ProbUnet_Second_v2.fcomb


ProbUnet_Third = ProbUNet_Third(
    model=model_second,
    prior_second=prior_second,
    posterior_second=posterior_second,
    fcomb_second=fcomb_second,
    prob_unet_second=ProbUnet_Second,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_44'
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Third.load_state_dict(model_weight)
ProbUnet_Third.eval()
ProbUnet_Third.requires_grad_(False)

#%% Obtain the modules: model_third, prior_third, posterior_third, fcomb_third

unet_v5 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(unet_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    unet_weight[k_new] = unet_weight.pop(k)
unet_v5.load_state_dict(unet_weight)

version_prev = None

ProbUnet_First_v5 = ProbUNet_First(
    model=unet_v5,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_34'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v5.load_state_dict(model_weight)
ProbUnet_First_v5.eval()
ProbUnet_First_v5.requires_grad_(False)

unet_v6 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(unet_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    unet_weight[k_new] = unet_weight.pop(k)
unet_v6.load_state_dict(unet_weight)

ProbUnet_First_v6 = ProbUNet_First(
    model=unet_v6,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_34'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v6.load_state_dict(model_weight)

model_first = ProbUnet_First_v6.model
prior_first = ProbUnet_First_v6.prior
posterior_first = ProbUnet_First_v6.posterior
fcomb_first = ProbUnet_First_v6.fcomb

ProbUnet_Second_v3 = ProbUNet_Second(
    model=model_first,
    prior_first=prior_first,
    posterior_first=posterior_first,
    fcomb_first=fcomb_first,
    prob_unet_first=ProbUnet_First,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_43'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Second_v3.load_state_dict(model_weight)
ProbUnet_Second_v3.eval()
ProbUnet_Second_v3.requires_grad_(False)

unet_v7 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(unet_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    unet_weight[k_new] = unet_weight.pop(k)
unet_v7.load_state_dict(unet_weight)

ProbUnet_First_v7 = ProbUNet_First(
    model=unet_v7,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_34'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v7.load_state_dict(model_weight)
ProbUnet_First_v7.eval()
ProbUnet_First_v7.requires_grad_(False)

unet_v8 = Unet(in_channels=1, 
            classes=1, 
            encoder_name = 'tu-resnest50d', 
            encoder_weights = 'imagenet')
model_weight = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/results/SIIM_pneumothorax_segmentation/version_14/checkpoints/best_model.ckpt'
unet_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
for k in list(unet_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    unet_weight[k_new] = unet_weight.pop(k)
unet_v8.load_state_dict(unet_weight)

ProbUnet_First_v8 = ProbUNet_First(
    model=unet_v8,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=version_prev
)

version_no = 'version_34'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_First_v8.load_state_dict(model_weight)

model_first_v8 = ProbUnet_First_v8.model
prior_first_v8 = ProbUnet_First_v8.prior
posterior_first_v8 = ProbUnet_First_v8.posterior
fcomb_first_v8 = ProbUnet_First_v8.fcomb

ProbUnet_Second_v4 = ProbUNet_Second(
    model=model_first_v8,
    prior_first=prior_first_v8,
    posterior_first=posterior_first_v8,
    fcomb_first=fcomb_first_v8,
    prob_unet_first=ProbUnet_First_v7,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_43'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Second_v4.load_state_dict(model_weight)

model_second = ProbUnet_Second_v4.model
prior_second = ProbUnet_Second_v4.prior
posterior_second = ProbUnet_Second_v4.posterior
fcomb_second = ProbUnet_Second_v4.fcomb

ProbUnet_Third_v2 = ProbUNet_Third(
    model=model_second,
    prior_second=prior_second,
    posterior_second=posterior_second,
    fcomb_second=fcomb_second,
    prob_unet_second=ProbUnet_Second,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_44'
root_dir = '/home/u/qqaazz800624/Probabilistic-Neural-Networks'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Third_v2.load_state_dict(model_weight)

model_third = ProbUnet_Third_v2.model
prior_third = ProbUnet_Third_v2.prior
posterior_third = ProbUnet_Third_v2.posterior
fcomb_third = ProbUnet_Third_v2.fcomb

#%%

ProbUnet_Fourth = ProbUNet_Fourth(
    model=model_third,
    prior_third=prior_third,
    posterior_third=posterior_third,
    fcomb_third=fcomb_third,
    prob_unet_third=ProbUnet_Third,
    optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=1e-5),
    task='binary',
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=4, T_mult=1),
    beta=beta,
    latent_dim=latent_dim,
    max_epochs=max_epochs,
    model_name= model_name,
    batch_size_train=batch_size_train,
    loss_fn=loss_fn,
    version_prev=None
)

version_no = 'version_45'
weight_path = f'results/SIIM_pneumothorax_segmentation/{version_no}/checkpoints/best_model.ckpt'
model_weight = torch.load(os.path.join(root_dir, weight_path), map_location="cpu")["state_dict"]
ProbUnet_Fourth.load_state_dict(model_weight)
ProbUnet_Fourth.eval()

#%%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_module = SIIMDataModule(batch_size_test=1, num_workers_test=2)
test_data_loader = data_module.test_dataloader()
dice_metric = DiceMetric(include_background=True, reduction='none', ignore_empty=False)
discreter = AsDiscrete(threshold=0.5)

ProbUnet_Fourth.to(device)

dice_scores = []

with torch.no_grad():
    for data in tqdm(test_data_loader):
        img, label = data['input'].to(device), data['target']
        prediction_outputs, prior_mu, prior_sigma = ProbUnet_Fourth.predict_step(img)
        stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
        stacked_samples = torch.sigmoid(stacked_samples)
        prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)
        dice_metric(y_pred=discreter(prediction_heatmap.cpu().unsqueeze(0).unsqueeze(0)), y=discreter(label.unsqueeze(0)))
        dice_score = dice_metric.aggregate().item()
        dice_scores.append(dice_score)
        dice_metric.reset()

print('Dice score: ', sum(dice_scores)/len(dice_scores))

with open(f'results/dice_scores_ProbUnet_step4.json', 'w') as file:
    json.dump(dice_scores, file)

#%%

# import json
# with open('../results/dice_scores_ProbUnet_BCELoss.json', 'r') as file:
#     dice_scores_ProbUnet_BCELoss = json.load(file)

# #%%
# import matplotlib.pyplot as plt

# # Plot histogram of dice_scores
# plt.hist(dice_scores_ProbUnet_BCELoss, bins=10, edgecolor='black')
# plt.xlabel('Dice Score')
# plt.ylabel('Frequency')
# plt.title('Histogram of Dice Scores')
# plt.show()


# #%% Single image dice evaluation

# import os
# from siim_dataset import SIIMDataset

# fold_no = 'testing'
# # Good: 6, 92, 522 Bad: 532, 484, 168
# # large mask: 92, 417, 492, 339, 132, 302
# # large-medium mask: 338, 377
# # medium mask: 107, 136
# # medium-small mask: 29, 412
# # small mask: 128, 184
# img_serial = 339
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# test_dataset = SIIMDataset(folds=[fold_no], if_test=True)

# image = test_dataset[img_serial]['input']  # shape: [1, 512, 512]
# mask = test_dataset[img_serial]['target']  # shape: [1, 512, 512]

# input_image = image.unsqueeze(0).to(device)
# ProbUnet_Fourth.to(device)
# with torch.no_grad():
#     prediction_outputs, prior_mu, prior_sigma = ProbUnet_Fourth.predict_step(input_image)

# stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
# stacked_samples = torch.sigmoid(stacked_samples)
# prediction_heatmap = stacked_samples.mean(dim = 0, keepdim=False)
# prediction_heatmap = prediction_heatmap.cpu()

# discreter = AsDiscrete(threshold=0.5)
# dice_metric = DiceMetric(include_background=True, reduction='mean')
# dice_metric(y_pred=discreter(prediction_heatmap.unsqueeze(0).unsqueeze(0)), y=discreter(mask.unsqueeze(0)))
# dice_score = dice_metric.aggregate().item()
# dice_metric.reset()

# print('Dice score: ', dice_score)

# #%% Prediction heatmap

# import matplotlib.pyplot as plt
# plt.imshow(prediction_heatmap.detach().numpy().T, 
#            cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Prediction heatmap: {fold_no}_{img_serial}')

# #%% Uncertainty heatmap

# stacked_samples = prediction_outputs['samples'].squeeze(1).squeeze(1)
# stacked_samples = torch.sigmoid(stacked_samples)
# uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim=False)

# import matplotlib.pyplot as plt

# plt.imshow(uncertainty_heatmap.cpu().detach().numpy().T, 
#            cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Heatmap of Epistemic Uncertainty: {fold_no}_{img_serial}')

# #%% uncertainty mask

# input_image = image.unsqueeze(0).to(device)
# ProbUnet_Fourth.to(device)
# with torch.no_grad():
#     prediction_outputs, prior_mu, prior_sigma = ProbUnet_Fourth.predict_step(input_image)
#     stacked_samples = torch.sigmoid(prediction_outputs['samples'])
#     uncertainty_heatmap = stacked_samples.var(dim = 0, keepdim = False)
#     uncertainty_heatmap = uncertainty_heatmap.squeeze(0).squeeze(0)

# #%%
# # Thresholding the uncertainty heatmap
# #quantile = torch.kthvalue(uncertainty_heatmap.flatten(), int(0.975 * uncertainty_heatmap.numel())).values.item()
# quantile = torch.quantile(uncertainty_heatmap, 0.975).item()
# #print("97.5th quantile of uncertainty_heatmap:", quantile)
# mask_uncertainty = torch.where(uncertainty_heatmap.cpu() > quantile, 
#                                mask.squeeze(0), 
#                                torch.zeros_like(mask.squeeze(0)))


# #%%
# import torch
# import matplotlib.pyplot as plt

# plt.imshow(mask_uncertainty.detach().numpy().T, 
#            cmap='plasma', aspect='auto')
# plt.colorbar()
# plt.title(f'Uncertainty mask: {fold_no}_{img_serial}')

#%%

# import json

# # 從 JSON 文件中讀取列表
# with open('../results/dice_scores_Unet.json', 'r') as file:
#     dice_scores_Unet = json.load(file)

# with open('../results/dice_scores_ProbUnet_BCELoss.json', 'r') as file:
#     dice_scores_ProbUnet_BCELoss = json.load(file)

# #%%

# import numpy as np
# dice_scores_Unet = np.array(dice_scores_Unet)
# dice_scores_ProbUnet_BCELoss = np.array(dice_scores_ProbUnet_BCELoss)
# combined_scores = np.column_stack((dice_scores_Unet, dice_scores_ProbUnet_BCELoss))
# combined_scores[10:20]

#%%
# import matplotlib.pyplot as plt

# # Draw histogram for dice_scores_ProbUnet_BCELoss
# plt.hist(dice_scores_ProbUnet_BCELoss, bins=10, edgecolor='black', alpha=0.4, label='ProbUnet_BCELoss')

# # Draw histogram for dice_scores
# plt.hist(dice_scores_Unet, bins=10, edgecolor='black', alpha=0.5, label='Unet')

# # Draw histogram for dice_scores_ProbUnet
# plt.hist(dice_scores_ProbUnet, bins=10, edgecolor='black', alpha=0.4, label='ProbUnet_DiceLoss')

# # # Draw histogram for dice_scores_segformer
# # plt.hist(dice_scores_segformer, bins=10, edgecolor='black', alpha=0.5, label='Segformer')

# # Add labels and title
# plt.xlabel('Dice Score')
# plt.ylabel('Frequency')
# plt.title('Histogram of Dice Scores')

# # Add legend
# plt.legend()

# # Show the histogram
# plt.show()

#%%