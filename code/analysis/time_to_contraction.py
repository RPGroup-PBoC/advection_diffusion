#%%
import sys
sys.path.insert(0,'../')
import os
import active_matter_pkg as amp
from skimage import io, filters, measure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
amp.viz.plotting_style()

root = '../../../data/active_stress/atpdilutions/'
directory_list = np.sort([directory for directory in os.listdir(root) if 'slide' in directory])

data_root = []
for directory in directory_list:
    if any('filename_order.csv' in dirname for dirname in os.listdir(os.path.join(root,str(directory)))):
        df_sorted = pd.read_csv(os.path.join(root, str(directory),'%s_filename_order.csv' %str(directory)),sep=',')
        first_file = df_sorted[df_sorted['order']==0]['filename'].values[0]
        data_root.append(first_file)
    else:
        data_root.append(os.path.join(root, directory))

#data_root = str([directory for directory in data_directory if '210518' in directory][0])

camera_crop = np.s_[150:1100,440:1490]
#%%
df = pd.DataFrame([])

for imgset_path in tqdm(data_root):
    mt_imgfiles, _, _, _, subdirectory = amp.io.tiff_walk(imgset_path, parse_channels=True)

    if len(subdirectory)>0:
        subdir_pb = subdirectory['frame' in subdirectory]
        imgset_path = os.path.join(imgset_path,subdir_pb)

    df_info = amp.io.parse_filename(imgset_path)
    num_pb = df_info['photobleach frame number'].values[0]

    if np.isnan(num_pb):
        num_pb = np.min([80, len(mt_imgfiles)-10])
    # Look before photobleaching has taken place (before aster formation)
    precontraction_files = mt_imgfiles[:num_pb+10]

    n=0
    equiv_diam = 0
    while (equiv_diam < 400) and (n<num_pb+10):
        if n == len(precontraction_files):
            equiv_diam = np.nan
            major_axis_length = np.nan
            minor_axis_length = np.nan
            n = np.nan
            break
        im = io.imread(precontraction_files[n])[camera_crop]

        im_gauss = filters.gaussian(im, sigma=5) * (2**16-1)

        thresh = filters.threshold_li(im_gauss)
        im_binary = (im_gauss > thresh)

        im_label = measure.label(im_binary)
        im_border = amp.image_processing.border_clear(im_label, edge=10)
        im_clean = amp.image_processing.remove_small(im_border, area_thresh=100)

        regprops = measure.regionprops_table(im_clean, intensity_image=im_gauss, 
                                            properties=('label','bbox','area', 
                                                        'centroid', 'equivalent_diameter',
                                                        'major_axis_length','minor_axis_length'))
        df_regprops = pd.DataFrame(regprops)

        if len(df_regprops)<1:
            n+=1
            continue
        else:
            equiv_diam = df_regprops[df_regprops['area']==df_regprops['area'].max()]['equivalent_diameter'].values[0]
            major_axis_length = df_regprops[df_regprops['area']==df_regprops['area'].max()]['major_axis_length'].values[0]
            minor_axis_length = df_regprops[df_regprops['area']==df_regprops['area'].max()]['minor_axis_length'].values[0]
            n+=1
    """
    fig, ax = plt.subplots(1,2,figsize=(16,8))
    if n<num_pb+10:
        ax[0].imshow(im, cmap='flag')
        ax[0].set_title('%i' %n)
        ax[1].imshow(im_clean==df_regprops[df_regprops['area']==df_regprops['area'].max()]['label'].values[0])
        plt.show()
    elif n == len(precontraction_files):
        plt.show()
    else:
        ax[0].imshow(io.imread(mt_imgfiles[0])[camera_crop], cmap='flag')
        ax[1].imshow(io.imread(mt_imgfiles[n-1])[camera_crop], cmap='flag')
        plt.show()"""
    _df = pd.DataFrame([[equiv_diam, major_axis_length, minor_axis_length,
                        n-1,df_info['time interval (s)'].values[0]]],
                        columns=['equivalent_diameter', 'major_axis_length', 
                                'minor_axis_length','frame_num','time interval (s)'])
    _df['filename'] = imgset_path
    df = pd.concat([df,_df], ignore_index=True)

#%%
df.to_csv('../../analyzed_data/time_to_contraction_ncd236_atp.csv', sep=',')
#%%
std = np.zeros(len(precontraction_files))
mean = np.zeros(len(precontraction_files))

for n in range(len(precontraction_files)):
    im = io.imread(precontraction_files[n])[camera_crop]
    std[n] = np.std(im)
    mean[n] = np.mean(im)

frames = np.arange(0, len(std), 1)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.scatter(frames, std/mean)
# %%
im = io.imread(precontraction_files[40])[camera_crop]
filters.try_all_threshold(im)
# %%
fig, ax = plt.subplots(1,1,figsize=(12,8))
ax.imshow(im, cmap='flag')
# %%
thresh = filters.threshold_li(im)
im_binary = (im > thresh)
#im_border = amp.image_processing.border_clear(im_binary, edge=3)

im_label = measure.label(im_binary)
im_border = amp.image_processing.border_clear(im_label, edge=20)
im_clean = amp.image_processing.remove_small(im_border, area_thresh=100)

regprops = measure.regionprops_table(im_clean, intensity_image=im, 
                                    properties=('label','bbox','area', 'centroid', 'equivalent_diameter'))
df_regprops = pd.DataFrame(regprops)

aster = df_regprops[df_regprops['area']==df_regprops['area'].max()]

im_clean[im_clean!=aster['label'].values[0]]=0
im_segment = im_clean/im_clean.max() * im
# %%
equiv_diam = np.zeros(len(precontraction_files))

for n in tqdm(range(len(precontraction_files))):
    im = io.imread(precontraction_files[n])[camera_crop]

    thresh = filters.threshold_li(im)
    im_binary = (im > thresh)

    im_label = measure.label(im_binary)
    im_border = amp.image_processing.border_clear(im_label, edge=10)
    im_clean = amp.image_processing.remove_small(im_border, area_thresh=100)

    regprops = measure.regionprops_table(im_clean, intensity_image=im, 
                                        properties=('label','bbox','area', 'centroid', 'equivalent_diameter'))
    df_regprops = pd.DataFrame(regprops)

    equiv_diam[n] = df_regprops[df_regprops['area']==df_regprops['area'].max()]['equivalent_diameter'].values[0]

diff = np.diff(equiv_diam)
plt.plot(diff, marker='o')
# %%
idx = np.where(diff==diff.max())
# %%
df = pd.read_csv('../analyzed_data/time_to_contraction.csv', sep=',')
# %%
cmap='gist_rainbow'
for f,d in df[df['major_axis_length']>700].groupby('filename'):
    n = d['frame_num'].values[0]
    fig, ax = plt.subplots(1,3,figsize=(24,8))
    if n>0:
        ax[0].imshow(io.imread(os.path.join(f,'DLPRed','img_%.9i_DLP_Red_000.tif' %(n-1)))[camera_crop],cmap=cmap, vmin=0)
    ax[1].imshow(io.imread(os.path.join(f,'DLPRed','img_%.9i_DLP_Red_000.tif' %n))[camera_crop],cmap=cmap, vmin=0)
    if n<len([tiff_files for tiff_files in os.listdir(os.path.join(f,'DLPRed')) if '.tif' in tiff_files])-1:
        ax[2].imshow(io.imread(os.path.join(f,'DLPRed','img_%.9i_DLP_Red_000.tif' %(n+1)))[camera_crop],cmap=cmap, vmin=0)
    plt.show()
# %%
for f,d in df[df['equivalent_diameter']>300].groupby('filename'):
    n = d['frame_num'].values[0]
    mt_imgfiles, _, _, subdirectory = amp.io.tiff_walk(f, parse_channels=True)
    im0 = io.imread(mt_imgfiles[n-1])[camera_crop]

    thresh = filters.threshold_li(im0)
    im_binary = (im0 > thresh)

    im_label = measure.label(im_binary)
    im_border = amp.image_processing.border_clear(im_label, edge=10)
    im_clean0 = amp.image_processing.remove_small(im_border, area_thresh=100)

    im = io.imread(mt_imgfiles[n])[camera_crop]

    thresh = filters.threshold_li(im)
    im_binary = (im > thresh)

    im_label = measure.label(im_binary)
    im_border = amp.image_processing.border_clear(im_label, edge=10)
    im_clean = amp.image_processing.remove_small(im_border, area_thresh=100)
    
    fig, ax = plt.subplots(2,2,figsize=(16,16))
    ax[0,0].imshow(im0)
    ax[0,1].imshow(im_clean0>0)
    ax[1,0].imshow(im)
    ax[1,1].imshow(im_clean>0)
    plt.show()

# %%
mt_imgfiles, _, _, subdirectory = amp.io.tiff_walk(data_root[13])

fig, ax = plt.subplots(1,2,figsize=(16,8))
ax[0].imshow(io.imread(mt_imgfiles[0])[camera_crop])
ax[1].imshow(io.imread(mt_imgfiles[df['frame_num'].values[13]])[camera_crop])
# %%
