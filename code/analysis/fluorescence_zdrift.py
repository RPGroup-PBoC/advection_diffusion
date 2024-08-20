#%%
import sys
sys.path.insert(0, '../')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, draw
import active_matter_pkg as amp
amp.viz.plotting_style()

filelist = '../../analyzed_data/analyzing_filenames.txt'
with open(filelist, 'r') as filestream:
    files = [line[:-1] for line in filestream if 'slide' in line]

root = '../../../data/active_stress/photobleach_data'
data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])
data_root = str([directory for directory in data_directory if '210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1' in directory][0])

mt_imgfiles, _, _, _, _ = amp.io.tiff_walk(data_root)

df_graticule = pd.read_csv('../../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
um_per_pxl = df_graticule['micron_per_pixel'].values[0]

# Includes dictionary for determining photobleach number for some data
numpb_dict = dict({'210426_slide1_lane1_pos1':34,
                '210426_slide1_lane1_pos2':31,
                '210426_slide1_lane2_pos1':87})

data_root, motortype = amp.io.identify_root(data_root)
_, _, mt_trimmed, _, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

if len(subdirectory)>0:

    if any('filename_order.csv' in filename for filename in os.listdir(data_root)):

        df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
        data_root = df_csv[df_csv['order']==1]['filename'].values[0]

df_info = amp.io.parse_filename(data_root)
num_pb = df_info['photobleach frame number'].values[0]

im = io.imread(mt_imgfiles[num_pb])
plt.imshow(im, cmap='flag')

# Draw circle
y_camcent, x_camcent = 620, 1000
im_disk = np.zeros(np.shape(im), dtype=np.uint8)
rr, cc = draw.disk([y_camcent, x_camcent], 125 / um_per_pxl, shape=np.shape(im))
im_disk[rr, cc] = 1
plt.imshow(im_disk, alpha=0.3)
#%%
total_intensity = np.zeros(20)
for n in range(len(total_intensity)):
    im = io.imread(mt_imgfiles[num_pb + n]) - 1920
    total_intensity[n] = np.sum(im * im_disk)

plt.plot(total_intensity)
# %%
# perform over all images
n_start = 0
n_tot = 20
offset = 1920
df_fluorescence = pd.DataFrame()
for file in files:
    try:
        data_root, motortype = amp.io.identify_root(file)
    except:
        print(file)
        continue
    _, _, mt_trimmed, _, subdirectory = amp.io.tiff_walk(data_root)

    if len(subdirectory)>0:

        if any('filename_order.csv' in filename for filename in os.listdir(data_root)):

            df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
            data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    
    df_info = amp.io.parse_filename(data_root)
    num_pb = df_info['photobleach frame number'].values[0]
    if np.isnan(num_pb):
        num_pb = numpb_dict[file]
    total_fluorescence = np.zeros((2, n_tot))
    total_fluorescence[0,:] = np.arange(0, n_tot, 1)
    for n in range(n_start, np.min([len(mt_trimmed), n_start + n_tot])):
        im = io.imread(mt_trimmed[n]) - offset
        total_fluorescence[1,n] = np.sum(im)

    df = pd.DataFrame(total_fluorescence.T, columns=('frame', 'total_intensity'))
    df['ATP (uM)'] = df_info['ATP (uM)'].values[0]
    df['time interval (s)'] = df_info['time interval (s)'].values[0]
    df['pluronic'] = df_info['pluronic'].values[0]
    df['normalized_fluorescence'] = total_fluorescence[1,:] / total_fluorescence[1,0]
    df['motortype'] = motortype
    df['filename'] = data_root
    df_fluorescence = pd.concat([df_fluorescence, df], ignore_index=True)

# %%
fig, ax = plt.subplots(1,1, figsize=(6,4))

for f,d in df_fluorescence.groupby('filename'):
    ax.scatter(d['frame'], d['normalized_fluorescence'], alpha=0.05,
               color='dodgerblue')
for t,d in df_fluorescence.groupby('frame'):
    ax.scatter(t, d['normalized_fluorescence'].median(), 
               facecolor='w', edgecolor='tomato')
ax.axhline(1.0, 0, 20, ls='--', lw=3, color='k', zorder=-1, label='normalized=1')
ax.set_xlim(-0.05, 20.05)
ax.set_ylim(0.5, 2.0)
ax.set_xlabel('frames after photobleaching', fontsize=20)
ax.set_ylabel('normalized total intensity\nwithin the activated region',
              fontsize=20)
ax.scatter(np.nan, np.nan, color='dodgerblue', label='experiment')
ax.scatter(np.nan, np.nan, facecolor='white', edgecolor='tomato', label='median')
ax.legend(loc=2, fontsize=12)
plt.savefig('../../figures/SIFigX_total_intensity_postpb_zoom.pdf',
            bbox_inches='tight', facecolor='white')
# %%
for params,df in df_fluorescence.groupby(['motortype', 'ATP (uM)', 'pluronic']):
    fig, ax = plt.subplots(1,1, figsize=(6,4))

    for f,d in df.groupby('filename'):
        ax.scatter(d['frame'], d['total_intensity']/d[d['frame']==0]['total_intensity'].values[0],
                   alpha=0.2, color='dodgerblue')
    for t,d in df.groupby('frame'):
        ax.scatter(t, d['normalized_fluorescence'].median(),
                   facecolor='white', edgecolor='tomato')
    ax.axhline(1.0, 0, 20, ls='--', lw=3, color='k', zorder=-1)
    ax.set_xlim(-0.05, 20.05)
    ax.set_xlabel('frame number', fontsize=20)
    ax.set_ylabel('normalized total intensity\nwithin the activated region', fontsize=20)
    ax.set_title('motortype=%s, ATP=%i µM, pluronic=%ix' %params)
    plt.show()
# %%
for f,d in df_fluorescence[df_fluorescence['motortype']=='k401'].groupby('filename'):
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.scatter(d['frame'], d['total_intensity']/d[d['frame']==0]['total_intensity'].values[0],
                alpha=0.2, color='dodgerblue')
    ax.axhline(1.0, 0, 20, ls='--', lw=3, color='k', zorder=-1)
    ax.set_xlim(-0.05, 20.05)
    ax.set_xlabel('frame number', fontsize=20)
    ax.set_ylabel('normalized total intensity\nwithin the activated region', fontsize=20)
    ax.set_title(f)
    plt.show()
# %%
