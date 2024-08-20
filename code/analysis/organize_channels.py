#%%
import sys
sys.path.insert(0, '../')
import os
import numpy as np

root = '../../../data/active_stress/k401bac_photobleach/230216_k401atp'

folders_in_root = os.listdir(root)
folders_in_root = [f for f in folders_in_root if '.DS_Store' not in f]

channels = ['DLPBlue', 'DLPRed', 'DLPYellow']
fileends = ['DLP_Blue_000.tif', 'DLP_Red_000.tif', 'DLP_Yellow_000.tif']
dict_channel = {f:c for f,c in zip(fileends, channels)}

for folders in folders_in_root:
    folder_path = os.path.join(root,folders)
    confirmed_channels = []
    try:
        files = os.listdir(os.path.join(folder_path,'Pos0'))
    except:
        continue

    for f in files:
        # find index of filename ending in fileends that matches in f
        idxlist = [idx for idx, fe in enumerate(fileends) if fe in f]

        if len(idxlist) == 1:

            idx = idxlist[0]

            if dict_channel[fileends[idx]] not in confirmed_channels:
                os.mkdir(os.path.join(folder_path,dict_channel[fileends[idx]]))
                confirmed_channels += [dict_channel[fileends[idx]]]

            os.rename(os.path.join(folder_path,'Pos0',f), os.path.join(folder_path,dict_channel[fileends[idx]],f))

# %%
