#%%
# This is a permanent deletion of all files for the All_Off channel. Images
# are saved as separate files to ensure that they have the proper file name
# ending ('All_Off_000.tif')
import sys
sys.path.insert(0, '../')
import os
import active_matter_pkg as amp

data_root = '../../../data/active_stress/mt647_mt488'
fileset = amp.io.find_all_tiffs(data_root)

alloff_set = [f for f in fileset if 'All_Off_000.tif' in f]
for f in alloff_set:
    os.remove(f)

# %%
