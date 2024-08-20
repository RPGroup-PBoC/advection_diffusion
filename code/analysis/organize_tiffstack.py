#%%
import os
import numpy as np
from skimage import io

root = '../../../data/active_stress/k401bac_photobleach/'
folders = np.sort(os.listdir(root))
folders = [f for f in folders if '.DS_Store' not in f]

for folder in folders:
    files = os.listdir(os.path.join(root,folder))
    file = [f for f in files if '.ome.tif' in f][0]
    if ('DLPRed' in file) and ('DLPRed' not in files):
        os.mkdir(os.path.join(root,folder,'DLPRed'))

    im = io.imread(os.path.join(root,folder,file))

    for n in range(np.shape(im)[0]):
        io.imsave(os.path.join(root,folder,'DLP_Red','img_%.9d_DLP_Red_000.tif' %n),im[n,:,:,1])
    os.remove(os.path.join(root,folder,file))
# %%
# Rename tiff files
root = '../../../data/active_stress/k401bac_photobleach/'
folders = np.sort(os.listdir(root))
folders = [f for f in folders if '.DS_Store' not in f]

for folder in folders:
    files = os.listdir(os.path.join(root,folder,'DLPRed'))
    files = [f for f in files if '.DS_Store' not in f]

    for file in files:
        if not 'DLP_Red_000.tif' in file:
            num_uscore = file.rfind('_')
            n = int(file[num_uscore+1:-4])
            os.rename(os.path.join(root,folder,'DLPRed',file),
                    os.path.join(root,folder,'DLPRed','img_%.9d_DLP_Red_000.tif' %n))
# %%
