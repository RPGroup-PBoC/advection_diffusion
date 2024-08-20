#%%
# This is a permanent deletion of all files for the DLP Blue channel. Images
# are saved as separate files to ensure that they have the proper file name
# ending ('DLP_Blue_000.tif'). Images can be recovered from the SpicyBeef 
# data storage in lab
import sys
sys.path.insert(0,'../')
import os
import active_matter_pkg as amp

data_root = '../../../data/active_stress/k401bac_photobleach/230216_k401atp'
fileset = amp.io.find_all_tiffs(data_root)

alloff_set = [f for f in fileset if 'DLP_Blue_000.tif' in f]
for f in alloff_set:
    os.remove(f)

# %%
