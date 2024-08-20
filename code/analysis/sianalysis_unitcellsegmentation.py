#%%
# Examining different segmentation schemes of unit cells
import sys
sys.path.insert(0, '../')
import os
import active_matter_pkg as amp
import numpy as np
import pandas as pd
from skimage import io, filters
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocess import Pool, cpu_count
tqdm.pandas()
amp.viz.plotting_style()

root = '../../../data/active_stress/photobleach_data'
data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])

data_root = str([directory for directory in data_directory if '210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1' in directory][0])

# %%
_, _, mt_trimmed, _, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

field = io.imread(mt_trimmed[0])
im = amp.image_processing.background_subtract(field, sigma=20, bitdepth=16)

# %%
fig, ax = plt.subplots(4,2, figsize=(8,16))
ax[0,0].imshow(field)
ax[0,0].set_title('raw')

ax[0,1].imshow(im)
ax[0,1].set_title('background subtracted')

thresh_options = ['isodata', 'li', 'mean', 'otsu', 'triangle', 'yen']

n = 0
for thresh_opt in thresh_options:
    if thresh_opt == 'isodata':
        thresh = filters.threshold_isodata(im)
    elif thresh_opt == 'li':
        thresh = filters.threshold_li(im)
    elif thresh_opt == 'mean':
        thresh = filters.threshold_mean(im)
    elif thresh_opt == 'otsu':
        thresh = filters.threshold_otsu(im)
    elif thresh_opt == 'triangle':
        thresh = filters.threshold_triangle(im)
    elif thresh_opt == 'yen':
        thresh = filters.threshold_yen(im)
    
    c = int(n % 2)
    r = 1 + int(n / 2)

    n += 1

    im_thresh = (im > thresh)
    im_filled = amp.image_processing.clean_unitcells(im_thresh, 
                                                     small_thresh=800, 
                                                     large_thresh=4000)

    ax[r,c].imshow(im_filled > 0)
    ax[r,c].set_title(thresh_opt)

for a in ax.ravel():
    a.set_xticklabels([])
    a.set_yticklabels([])

#plt.savefig('../../figures/SIFigX_thresholding_schemes.pdf',
#            bbox_inches='tight', facecolor='white')
# %%
