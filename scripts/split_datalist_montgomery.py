#%%

import json
import os
from glob import glob

import pydicom
from sklearn.model_selection import KFold
from tqdm import tqdm


#%%

data_root = '/home/u/qqaazz800624/Probabilistic-Neural-Networks/data/MontgomerySet'
image_dir = os.path.join(data_root, 'CXR_png')
label_dir = os.path.join(data_root, 'ManualMask')

label_dir

#%%





#%%