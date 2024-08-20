#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, scaling
import openpiv.filters
from skimage import io, morphology, feature, filters, segmentation, measure, transform, draw, color
from scipy import optimize, signal
import bokeh as bk
import active_matter_pkg as amp
amp.viz.plotting_style()

data_root = '../../../data/active_stress/210426_ncd_photobleach/210426_slide1_lane2_pos3_10s_intervals_10ms_iLidmicroNcd_Alloff_200ms_DLPYellow_200ms_DLPRed_50ms_DLPBlue_skip1_frame100_photobleach_1'
fileset = amp.io.find_all_tiffs(data_root)

num_skipstr = data_root.find('skip')
num_uscore = num_skipstr + data_root[num_skipstr:].find('_')
num_skip = int(data_root[num_skipstr+4:num_uscore])

if 'frame' in data_root:
    num_frame = data_root.find('frame')
    num_uscore = num_frame + data_root[num_frame:].find('_')
    # Photobleaching occurs prior to the activation cycle listed as frame## in the filename
    # Then there is indexing by 0 in python, thus subtracting by 2
    num_pb = 2 * (int(data_root[num_frame+5:num_uscore]) - 1) - 1

df_graticule = pd.read_csv('../analyzed_data/objective_pxl_micron_scale.csv', sep=',')

mt_imgfiles = np.sort([imfile for imfile in fileset if 'DLP_Red_000.tif' in imfile])
mot_imgfiles = np.sort([imfile for imfile in fileset if 'DLP_Yellow_000.tif' in imfile])

im_first = io.imread(mt_imgfiles[0])
im_before_pb = io.imread(mt_imgfiles[num_pb-1])
# At this time, the first two images taken after photobleaching are
# taken with little interval in between. Skipping the first of the two
im_pb1 = io.imread(mt_imgfiles[num_pb+1])
im_pb2 = io.imread(mt_imgfiles[num_pb+2])

window = np.s_[200:1100,500:1400]
fig, ax = plt.subplots(2,2,figsize=(16,16))
ax[0,0].imshow(im_before_pb[window])
ax[0,1].imshow(io.imread(mt_imgfiles[num_pb])[window])
ax[1,0].imshow(im_pb1[window])
ax[1,1].imshow(im_pb2[window])
#%%
# Attempt background subtract using first image. Then set values below 0 to 0
thresh = filters.threshold_otsu(im_before_pb)
im_binary = (im_before_pb > thresh)
im_thresh = im_before_pb * im_binary
#im_bksub[im_bksub<0] = 0

fig, ax = plt.subplots(2,2,figsize=(16,16))
ax[0,0].imshow(im_before_pb, cmap='flag')
ax[0,1].imshow(im_binary)
ax[1,0].imshow(im_thresh)
#%%
im_blur = filters.gaussian(im_thresh, sigma=50) * (2**16 - 1)
# Guess parameters
alpha_guess = im_blur.max()
if alpha_guess == 2**16 - 1:
    Warning('Camera may be saturated')
y_guess = np.where(im_blur==alpha_guess)[0][0]
x_guess = np.where(im_blur==alpha_guess)[1][0]

winsize = 150
crop_window = np.s_[y_guess-winsize:y_guess+winsize,x_guess-winsize:x_guess+winsize]

y = np.arange(np.shape(im_blur[crop_window])[0], step=1)
x = np.arange(np.shape(im_blur[crop_window])[1], step=1)
X, Y = np.meshgrid(x,y)
coords = np.column_stack((np.ravel(X), np.ravel(Y)))

sigma_xguess, sigma_yguess = 20, 20
p0 = [150, 150, sigma_xguess, sigma_yguess, alpha_guess]
popt, pcov = optimize.curve_fit(amp.image_processing.gaussian_2d, coords, np.ravel(im_blur[crop_window]), p0=p0)

popt[0] += x_guess - winsize
popt[1] += y_guess - winsize

max_sigma = popt[2:4].max()

window = np.s_[int(popt[1]-4*max_sigma):int(popt[1]+4*max_sigma),int(popt[0]-4*max_sigma):int(popt[0]+4*max_sigma)]

im_prepb = im_before_pb[window]
im_postpb1 = im_pb1[window]
im_postpb2 = im_pb2[window]
im_postpb4 = io.imread(mt_imgfiles[num_pb+4])[window]
# %%
cmap='flag'

fig, ax = plt.subplots(2,2,figsize=(16,16))
ax[0,0].imshow(im_prepb, cmap=cmap)
ax[0,0].set_title('before photobleaching', fontsize=16)

ax[0,1].imshow(im_postpb1, cmap=cmap, vmax=im_postpb1.max())
ax[0,1].set_title('after photobleaching', fontsize=16)

ax[1,0].imshow(im_postpb2, cmap=cmap, vmax=im_postpb2.max())
ax[1,0].set_title('2 frames after photobleaching', fontsize=16)

ax[1,1].imshow(im_postpb4, cmap=cmap, vmax=im_postpb4.max())
ax[1,1].set_title('4 frames after photobleaching', fontsize=16)

# %%
intensity = np.zeros((len(mt_imgfiles[num_pb+1:]),2))
for n_postpb in range(len(mt_imgfiles[num_pb+1:])):
    im = io.imread(mt_imgfiles[num_pb+1+n_postpb])[window]
    intensity[n_postpb,0] = n_postpb
    intensity[n_postpb,1] = np.sum(im)

plt.scatter(intensity[:,0],intensity[:,1],color='rebeccapurple')
plt.xlim([0,30])
plt.xlabel('frames after photobleaching', fontsize=16)
plt.ylabel('total pixel intensity',fontsize=16)
#plt.savefig('../figures/pixel_intensity.pdf', bbox_inches='tight',
#            facecolor='white')
# %%
# Break up into individual cells
# Trying simple segmentation
im_relabel, regionprops = amp.image_processing.process_aster_cells(im_postpb1)
im_relabel2, regionprops2 = amp.image_processing.process_aster_cells(im_postpb2)

# %%
list_centroids = np.zeros((len(regionprops),5))
list_centroids[:,0:3] = regionprops[['label','centroid-0','centroid-1']].values

average_xdiff = np.average(list_centroids[1:,1] - list_centroids[:-1,1])
average_ydiff = np.average(np.sort(list_centroids[:,2])[1:] - np.sort(list_centroids[:,2])[:-1])

row_counter = 0
list_centroids[0,3] = row_counter
for n in range(len(regionprops)-1):
    if list_centroids[n+1,1] - list_centroids[n,1] > average_xdiff:
        row_counter += 1
        list_centroids[n+1,3] = row_counter
    else:
        list_centroids[n+1,3] = row_counter

col_counter = 0
# Reorder list for counting
list_centroids = list_centroids[list_centroids[:,2].argsort()]
list_centroids[0,4] = col_counter
for n in range(len(regionprops)-1):
    if list_centroids[n+1,2] - list_centroids[n,2] > average_ydiff:
        col_counter += 1
        list_centroids[n+1,4] = col_counter
    else:
        list_centroids[n+1,4] = col_counter

# Return order to indexing
list_centroids = list_centroids[list_centroids[:,0].argsort()]

# %%
if len(np.unique(im_relabel))!=len(np.unique(im_relabel2)):
    Warning('Mismatch in successive frames between images')

crop_ext = 10
square_grid = int(np.ceil(np.sqrt(len(regionprops))))

fig, ax = plt.subplots(square_grid,square_grid,figsize=(12,12))
for n_cell in range(1,im_relabel.max()+1):
    row = int(list_centroids[n_cell-1,3])
    col = int(list_centroids[n_cell-1,4]-1)
    ax[row, col].imshow(im_postpb1 * (im_relabel==n_cell), cmap='Reds', alpha=0.75)
    ax[row, col].imshow(im_postpb2 * (im_relabel2==n_cell), cmap='Blues', alpha=0.76)
    bbox = regionprops[regionprops['label']==n_cell][['bbox-0','bbox-1','bbox-2','bbox-3']].values[0]
    ax[row, col].set_ylim([bbox[2]+crop_ext, bbox[0]-crop_ext])
    ax[row, col].set_xlim([bbox[1]-crop_ext, bbox[3]+crop_ext])
    ax[row, col].set_title('%i' %n_cell)
    ax[row, col].set_xticklabels([])
    ax[row, col].set_yticklabels([])
    ax[row, col].set_facecolor((0,0,0))
#%%
fig, ax = plt.subplots(square_grid,square_grid,figsize=(12,12))
for n_cell in range(1,im_relabel.max()+1):
    row = int(list_centroids[n_cell-1,3])
    col = int(list_centroids[n_cell-1,4])
    ax[row, col].imshow(im_postpb1 * (im_relabel==n_cell), cmap='Reds', alpha=1)
    bbox = regionprops[regionprops['label']==n_cell][['bbox-0','bbox-1','bbox-2','bbox-3']].values[0]
    ax[row, col].set_ylim([bbox[2]+crop_ext, bbox[0]-crop_ext])
    ax[row, col].set_xlim([bbox[1]-crop_ext, bbox[3]+crop_ext])
    ax[row, col].set_xticklabels([])
    ax[row, col].set_yticklabels([])
    ax[row, col].set_facecolor((0,0,0))
#%%
fig, ax = plt.subplots(square_grid,square_grid,figsize=(12,12))
for n_cell in range(1,im_relabel.max()+1):
    row = int(list_centroids[n_cell-1,3])
    col = int(list_centroids[n_cell-1,4])
    ax[row, col].imshow(im_postpb2 * (im_relabel==n_cell), cmap='Blues', alpha=1)
    bbox = regionprops[regionprops['label']==n_cell][['bbox-0','bbox-1','bbox-2','bbox-3']].values[0]
    ax[row, col].set_ylim([bbox[2]+crop_ext, bbox[0]-crop_ext])
    ax[row, col].set_xlim([bbox[1]-crop_ext, bbox[3]+crop_ext])
    ax[row, col].set_xticklabels([])
    ax[row, col].set_yticklabels([])
    ax[row, col].set_facecolor((0,0,0))
#%%
df = pd.DataFrame([])
for n in range(5):
    im = io.imread(mt_imgfiles[num_pb+n])[window]
    im_label, regprops = amp.image_processing.process_aster_cells(im)
    regprops['frames_after_photobleach'] = n

    df = df.append(regprops,ignore_index=True)

# %%
for n in range(6):
    fig, ax = plt.subplots(1,2,figsize=(16,8))
    im = io.imread(mt_imgfiles[num_pb+n-1])[window]
    im_blur = filters.gaussian(im, sigma=100)
    im_subt = im - im_blur * (2**16 - 1)
    thresh = filters.threshold_mean(im_subt)
    im_bin = (im_subt > thresh)
    im_thresh = im_bin * im
    ax[0].imshow(im)
    ax[1].imshow(im_thresh)
#%%
length = int(3.41*25)
cushion = 5
fig, ax = plt.subplots(1,4,figsize=(16,4))
for n in range(4):
    im = io.imread(mt_imgfiles[num_pb+3*n-1])[window]
    if n==3:
        im[int(-1 * (cushion + 5)):int(-1 * cushion),int(-1 * (length + cushion)):int(-1 * cushion)] = im.max()
    ax[n].imshow(im)
    ax[n].set_xticklabels([])
    ax[n].set_yticklabels([])
    if n==0:
        ax[n].set_title('before photobleach')
    else:
        ax[n].set_title('%i minutes after pb' %(n))
#plt.savefig('../../../presentations/collection/frames_after_pb.pdf',
#            bbox_inches='tight',background_color='white')
# %%
n=1
im = io.imread(mt_imgfiles[num_pb+n])[window]
im_blur = filters.gaussian(im, sigma=100)
im_subt = im - im_blur * (2**16 - 1)
thresh = filters.threshold_mean(im_subt)
im_bin = (im_subt > thresh)
im_thresh = im_bin * im
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.imshow(im)
plt.savefig('../../../presentations/collection/image_example.pdf',
            bbox_inches='tight',background_color='white')

# %%
n_slice = 180
fig, ax = plt.subplots(1,2,figsize=(16,8))
for n in range(12):
    im = io.imread(mt_imgfiles[num_pb+n-1])[window]
    im_slice = np.copy(im[:,n_slice])
    im_norm = (im_slice - im_slice.min()) / (im_slice.max() - im_slice.min())
    im_blur = filters.gaussian(im, sigma=100)
    im_subt = im - im_blur * (2**16 - 1)
    thresh = filters.threshold_mean(im_subt)
    im_bin = (im_subt > thresh)
    im_thresh = im_bin * im
    im_thresh_slice = np.copy(im_thresh[:,n_slice])
    im_thresh_norm = (im_thresh_slice - im_thresh_slice.min()) / (im_thresh_slice.max() - im_thresh_slice.min())
    ax[1].plot(im_norm - 0.5*n, lw=3)
    ax[1].set_title('offset normalized intensity profiles', fontsize=16)
    ax[1].set_yticklabels([])
    if n==0:
        ax[0].plot(im_norm - 0.5*n, lw=3, label='before pb')
    else:
        ax[0].plot(im_norm - 0.5*n, lw=3, label='%i after pb' %(n+1))
    ax[0].set_title('normalized intensity profiles', fontsize=16)
    ax[0].set_yticklabels([])
ax[0].legend()
plt.savefig('../../../presentations/collection/aster_intensity_line_profiles.pdf')
#ax[0].set_xlim([330,600])
#ax[1].set_xlim([330,600])
# %%
cmap='Greys_r'
fig, ax = plt.subplots(2,2,figsize=(16,16))
ax[0,0].imshow(io.imread(mt_imgfiles[num_pb-1])[window],cmap=cmap)
ax[0,1].imshow(io.imread(mt_imgfiles[num_pb+10])[window],cmap=cmap)
ax[1,0].imshow(io.imread(mt_imgfiles[num_pb+20])[window],cmap=cmap)
ax[1,1].imshow(io.imread(mt_imgfiles[num_pb+30])[window],cmap=cmap)

# %%
fig, ax = plt.subplots(1,1,figsize=(8,8))
for n in range(len(df[df['label']==7])):
    ax.scatter(df[(df['frames_after_photobleach']==n)]['centroid-1'].values,
                df[(df['frames_after_photobleach']==n)]['centroid-0'].values,
                label='%i frames after photobleaching' %n)
ax.legend()
# %%
df_intensity = df[['label','frames_after_photobleach']].copy()
df_intensity['total_intensity'] = np.zeros(len(df))
for l, d in df.groupby(['label','frames_after_photobleach']):
    df_intensity.loc[(df_intensity['label']==l[0]) & (df_intensity['frames_after_photobleach']==l[1]),'total_intensity'] = np.sum(d['intensity_image'].values[0])

df = pd.merge(df,df_intensity, on=['label','frames_after_photobleach'])
#%%
fig, ax = plt.subplots(1,1,figsize=(8,8))
for label, d in df.groupby('label'):
    ax.plot(d['frames_after_photobleach'],d['total_intensity'],
            lw=2, marker='.', markersize=20, label='%ith unit cell' %label)
ax.legend()
# %%
# Examining intensity within a cropped window as a function of time
intensity_postpb = np.zeros((2, len(mt_imgfiles) - num_pb - 1))
for n in range(len(mt_imgfiles) - num_pb - 1):
    im = io.imread(mt_imgfiles[num_pb + n + 1])[window]
    intensity_postpb[0,n] = 1 + n
    intensity_postpb[1,n] = np.sum(im)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.scatter(intensity_postpb[0,:], intensity_postpb[1,:])
# %%
correlation2d = signal.correlate2d(im_pb1[window], im_pb2[window],
                                        mode='same', boundary='fill', 
                                        fillvalue=0)
y, x = np.unravel_index(np.argmax(correlation2d), correlation2d.shape)
# %%
im1 = im_pb1[window].copy()
im2 = im_pb2[window].copy()

im1 = (im1 - im1.min()) / (im1.max() - im1.min())
im2 = (im2 - im2.min()) / (im2.max() - im2.min())

correlation2d = signal.correlate2d(im1, im2, mode='same', 
                                    boundary='fill', fillvalue=0)
y, x = np.unravel_index(np.argmax(correlation2d), correlation2d.shape)

fig, ax = plt.subplots(1, 3, figsize=(12,8))

ax[0].imshow(im1)
ax[0].scatter(x, y, color='tomato', s=20)
ax[0].set_axis_off()

ax[1].imshow(im2)
ax[1].set_axis_off()

ax[2].imshow(correlation2d)
# %%
fig, ax = plt.subplots(1,2,figsize=(16,8))
im = io.imread(mt_imgfiles[num_pb+12])[window]
im_blur = filters.gaussian(im, sigma=100)
im_subt = im - im_blur * (2**16 - 1)
thresh = filters.threshold_mean(im_subt)
im_bin = (im_subt > thresh)
im_thresh = im_bin * im
ax[0].imshow(im)
ax[1].imshow(im_bin)
# %%
fig, ax = plt.subplots(2,2,figsize=(16,16))
for n in range(4):
    im = io.imread(mt_imgfiles[num_pb+2*n+2])[window]
    if n==3:
        im[350:355,int(355-40*df_graticule['pxl_per_micron'].values[0]):355] = im.max()
    row = int(n / 2)
    col = int(n % 2)
    ax[row,col].imshow(im)
    ax[row,col].set_xticklabels([])
    ax[row,col].set_yticklabels([])
    if n==0:
        ax[row,col].set_title('immediately after photobleaching', fontsize=16)
    else:
        ax[row,col].set_title('%i seconds after photobleaching' %(n*40), fontsize=16)
    if n==3:
        ax[row,col].text(int(355-20*df_graticule['pxl_per_micron']),340,'40 µm',
                ha='center', va='center', color='white', fontsize=14)
plt.savefig('../figures/aster_formation_2x2.pdf', bbox_inches='tight',
            facecolor='white')
# %%
fig, ax = plt.subplots(2,4,figsize=(24,12))
for n in range(8):
    im = io.imread(mt_imgfiles[num_pb+3*n+2+9])[window]
    if n==7:
        im[350:355,int(355-40*df_graticule['pxl_per_micron'].values[0]):355] = im.max()
    row = int(n / 4)
    col = int(n % 4)
    ax[row,col].imshow(im)
    ax[row,col].set_xticklabels([])
    ax[row,col].set_yticklabels([])
    ax[row,col].set_title('%i minutes after photobleaching' %(3+n), fontsize=16)
    if n==7:
        ax[row,col].text(int(355-20*df_graticule['pxl_per_micron']),340,'40 µm',
                ha='center', va='center', color='white', fontsize=14)
plt.savefig('../figures/aster_late_formation_2x4.pdf', bbox_inches='tight',
            facecolor='white')
# %%
