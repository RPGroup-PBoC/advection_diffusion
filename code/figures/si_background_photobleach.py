#%%
import sys
sys.path.insert(0, '../')
import os
import numpy as np
from skimage import io, filters
import matplotlib.pyplot as plt
import active_matter_pkg as amp
import pandas as pd
amp.viz.plotting_style()

root = '../../../data/active_stress/control_experiments/mt_projector_photobleach/02-13-2022_Photobleach_NcdYFP_NoMT_EM577-690_1/'
data_dir = '02-11-2022_Photobleach_NcdYFP_MT2_EM577-690_2'
data_dir_noMT = '02-13-2022_Photobleach_NcdYFP_NoMT_EM577-690_1'
imgfiles = amp.io.find_all_tiffs(os.path.join(root,data_dir))
noMTimgs = amp.io.find_all_tiffs(os.path.join(root,data_dir_noMT))
# %%
offset = 1920
im = io.imread(imgfiles[0])
window = np.s_[300:900,660:1260]
plt.imshow(im[window])
# %%
dt = 10
time = np.arange(0,dt*len(imgfiles),dt)
total_intensity = np.zeros(len(imgfiles))
mean_intensity = np.zeros(len(imgfiles))
for n in range(len(imgfiles)):
    im = io.imread(imgfiles[n])[window] - offset
    total_intensity[n] = np.sum(im)
    mean_intensity[n] = np.mean(im)
# %%
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].scatter(time/60, total_intensity/total_intensity.max(),
              color='dodgerblue')
ax[1].scatter(time/60, mean_intensity/mean_intensity.max(),
              color='dodgerblue')

for a in ax:
    a.set_xlabel('time [min]', fontsize=20)
    a.set_ylim([0.5,1.5])
ax[0].set_ylabel(r'total intensity (normalized to $t=0$)', fontsize=20)
ax[1].set_ylabel(r'mean intensity (normalized to $t=0$)', fontsize=20)
plt.savefig('../figures/SIFig_projector_bleaching.pdf', bbox_inches='tight',
            facecolor='white')
# %%
# method 2 is to image in smaller field
dt = 10
time = np.arange(0,dt*len(imgfiles),dt)
total_intensity_windowed = np.zeros(len(imgfiles))
mean_intensity_windowed = np.zeros(len(imgfiles))
std_intensity_windowed = np.zeros(len(imgfiles))

for n in range(len(imgfiles)):
    im = io.imread(imgfiles[n]) - offset
    mean_intensity_windowed[n] = np.mean(im[window])
    std_intensity_windowed[n] = np.std(im[window])

time_nomt = np.arange(0,dt*len(noMTimgs),dt)
mean_nomt_windowed = np.zeros(len(noMTimgs))
std_nomt_windowed = np.zeros(len(noMTimgs))

for n in range(len(noMTimgs)):
    im = io.imread(noMTimgs[n]) - offset
    mean_nomt_windowed[n] = np.mean(im[window])
    std_nomt_windowed[n] = np.std(im[window])

# %%
fig, ax = plt.subplots(1,2,figsize=(14,6))
ax[0].scatter(time/60,mean_intensity_windowed/mean_intensity_windowed[0],
            color='dodgerblue',alpha=0.5)
ax[0].fill_between(time/60,(mean_intensity_windowed-std_intensity_windowed)/mean_intensity_windowed[0],
                (mean_intensity_windowed+std_intensity_windowed)/mean_intensity_windowed[0],color='dodgerblue',
                alpha=0.3)
ax[1].scatter(time_nomt/60,mean_nomt_windowed/mean_nomt_windowed[0],
            color='dodgerblue',alpha=0.5)

for a in ax:
    a.set_xlabel('time [min]', fontsize=20)
ax[0].set_ylim([0.5,1.55])
ax[1].set_ylim([0.0, 2.1])

ax[0].set_ylabel(r'mean intensity (normalized to $t=0$)', fontsize=20)
ax[1].set_ylabel(r'mean intensity (normalized to $t=0$)', fontsize=20)
ax[0].set_xlim([-1,time[-1]/60+1])

ax[1].set_xlim([-1,time_nomt[-1]/60+1])
ax[0].set_title('with microtubules and\nno motor dimerization', fontsize=20)
ax[1].set_title('without microtubules', fontsize=20)
ax[0].text(-3, 1.6, '(A)', ha='right', va='bottom', fontsize=20)
ax[1].text(-3, 2.2, '(B)', ha='right', va='bottom', fontsize=20)

plt.savefig('../../figures/SIFig_projector_bleaching_windowed.pdf', bbox_inches='tight',
            facecolor='white')
# %%
