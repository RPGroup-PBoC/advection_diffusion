#%%
import sys
sys.path.insert(0, '../')
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, filters, feature
from skimage.morphology import disk
import pandas as pd
import active_matter_pkg as amp
amp.viz.plotting_style()

root = '../../../data/active_stress/mt647_mt488/220614_ncd_mt647_mt488/'
file = '06-14-2022_slide2_lane2_pos5_10s_intervals_8thcircle_NcdYFP_Mt647_Mt488_10ms_AllOff_100ms_DLPRed_75ms_DLPBlue_skip0_frame70_photobleach_1'

data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])
data_root = str([directory for directory in data_directory if file in directory][0])

imgfiles = amp.io.tiff_walk(data_root, parse_channels=False)
mt_red = np.sort([file for file in imgfiles[0] if '/DLPRed/' in file])
mt_blue = np.sort([file for file in imgfiles[0] if '/DLPBlue/' in file])

df_info = amp.io.parse_filename(data_root)
num_pb = df_info['photobleach frame number'].values[0] + 1

win = np.s_[325:875,700:1250]

im_first = io.imread(mt_red[0])[win]
im_before_pb = io.imread(mt_red[num_pb-1])[win]
# At this time, the first two images taken after photobleaching are 
# taken with little interval in between. Skipping the first of the two
im_pb1 = io.imread(mt_red[num_pb])[win]
im_pb2 = io.imread(mt_red[num_pb+1])[win]

im_blue_before_pb = io.imread(mt_blue[num_pb-1])[win]
# At this time, the first two images taken after photobleaching are 
# taken with little interval in between. Skipping the first of the two
im_blue_pb1 = io.imread(mt_blue[num_pb])[win]
im_blue_pb2 = io.imread(mt_blue[num_pb+1])[win]

cmap='gray'
fig, ax = plt.subplots(3,2,figsize=(12,18))
ax[0,0].imshow(im_before_pb, cmap=cmap)
ax[1,0].imshow(im_pb1, cmap=cmap)
ax[2,0].imshow(im_pb2, cmap=cmap)

ax[0,1].imshow(im_blue_before_pb, cmap=cmap)
ax[1,1].imshow(im_blue_pb1, cmap=cmap)
ax[2,1].imshow(im_blue_pb2, cmap=cmap)
    # %%
cmap = 'flag'
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].imshow(im_before_pb, cmap=cmap)
ax[1].imshow(im_blue_before_pb, cmap=cmap)

# %%
nc, nr = np.shape(im_before_pb)
if 'overlay' not in os.listdir(data_root):
    os.mkdir(os.path.join(data_root,'overlay'))
# Segment circular region
for n in range(num_pb-1, len(mt_blue)):
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    overlay = np.zeros((nc, nr, 3))
    im = io.imread(mt_blue[n])
    im_red = io.imread(mt_red[n])
    im_rgauss = filters.gaussian(im_red, sigma=30)
    im_subt = im_red - im_rgauss
    im_win = im_subt[win]

    thresh = filters.threshold_li(im)
    im_thresh = im * (im > thresh)
    im_nan = im_thresh.copy().astype('float')
    im_nan[im_nan==0] = np.nan

    im_bluenorm = (im_nan[win] - np.nanmin(im_nan[win])) / (np.nanmax(im_nan[win]) - np.nanmin(im_nan[win]))
    im_blue = np.nan_to_num(im_bluenorm)
    im_rednorm = (im_win - np.nanmin(im_win)) / (np.nanmax(im_win) - np.nanmin(im_win))

    overlay[..., 0] = im_rednorm
    overlay[..., 1] = 0
    overlay[..., 2] = 3 * im_blue / 4
    ax.imshow(overlay)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(os.path.join(data_root, 'overlay', 'im_%.03d.tiff' %n), bbox_inches='tight', facecolor='white')
# %%
if 'two_channels_colored' not in os.listdir(data_root):
    os.mkdir(os.path.join(data_root,'two_channels_colored'))
# Segment circular region
for n in range(num_pb-1, len(mt_blue)):
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    im = io.imread(mt_blue[n])
    nc, nr = np.shape(im[win])
    blue_channel = np.zeros((nc, nr, 3))
    red_channel = np.zeros((nc, nr, 3))
    thresh = filters.threshold_li(im)
    im_thresh = im * (im > thresh)
    im_nan = im_thresh.copy().astype('float')
    im_nan[im_nan==0] = np.nan
    blue_channel[...,2] = amp.image_processing.normalize(im_nan[win])
    blue_channel[...,1] = amp.image_processing.normalize(im_nan[win]) / 4 * 3
    red_channel[...,0] = amp.image_processing.normalize(io.imread(mt_red[n])[win])
    red_channel[...,1] = amp.image_processing.normalize(io.imread(mt_red[n])[win]) / 2
    ax[0].imshow(red_channel)
    ax[1].imshow(blue_channel)
    ax[0].set_title('Alexa 647 (red)', fontsize=24)
    ax[1].set_title('Alexa 488 (cyan)', fontsize=24)
    for a in ax.flatten():
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
    plt.savefig(os.path.join(data_root, 'two_channels_colored', 'im_%.03d.tiff' %n), bbox_inches='tight', facecolor='white')

# %%
im = io.imread(mt_blue[num_pb-1])
selem = disk(3)
im_med = filters.median(im, selem=selem)
thresh = filters.threshold_li(im_med)
im_binary = (im > thresh)
im_thresh = im_binary * im
im_nan = im_thresh.copy().astype('float')
im_nan[im_nan==0] = np.nan
im_red = io.imread(mt_red[num_pb-1])
im_rednan = (im_red * im_binary).astype('float')
im_rednan[im_rednan==0] = np.nan
im_redwin = im_rednan[win]
im_nanwin = im_nan[win]
#%%
plt.figure(figsize=(8,8))
plt.scatter(im_redwin[np.logical_not(np.isnan(im_redwin))], im_nanwin[np.logical_not(np.isnan(im_nanwin))],
            alpha=0.01, color='dodgerblue', rasterized=True)

# %%

