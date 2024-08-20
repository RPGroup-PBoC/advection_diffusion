#%%
# Try looking at correlation length
import os

from numpy.lib.function_base import kaiser
import active_matter_pkg as amp
from skimage import io, filters, feature, morphology, measure, registration
from scipy import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from active_matter_pkg import image_processing
amp.viz.plotting_style()
%matplotlib inline

root = '../../../data/active_stress/photobleach_data'
data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])

data_root = str([directory for directory in data_directory if '210520_slide1_lane2_pos2' in directory][0])
#data_root = str([directory for directory in data_directory if '210519_slide2_lane2_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1' in directory][0])
#data_root = str([directory for directory in data_directory if '2018' in directory][0])

mt_imgfiles, mot_imgfiles, mt_trimmed, mot_trimmed, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

if len(subdirectory)>0:
    if any('filename_order.csv' in filename for filename in os.listdir(data_root)):
        df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
        data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    if any('image_crop.csv' in filename for filename in os.listdir(data_root)):
        df_crop = pd.read_csv(os.path.join(data_root,'image_crop.csv'), sep=',')
df_info = amp.io.parse_filename(data_root)
df_graticule = pd.read_csv('../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
um_per_pxl = df_graticule['micron_per_pixel'].values[0]

df_t2c = pd.read_csv('../analyzed_data/time_to_contraction.csv', sep=',')
#n_contract = df_t2c[df_t2c['filename']==data_root]['frame_num'].values[0]
#mt_trunc = [file for file in mt_imgfiles if data_root in file]
#num_pb = 53
num_pb = df_info['photobleach frame number'].values[0]
im_first = io.imread(mt_imgfiles[0])
im_before_pb = io.imread(mt_imgfiles[num_pb-1])
# At this time, the first two images taken after photobleaching are 
# taken with little interval in between. Skipping the first of the two
im_pb1 = io.imread(mt_imgfiles[num_pb+1])
im_pb2 = io.imread(mt_imgfiles[num_pb+2])

dict_color = dict({-1:'green',0:'rebeccapurple',1:'tomato',2:'dodgerblue'})

# bounding box restricting primarily to activation region
x_camcent = 610
y_camcent = 975
if num_pb < 60:
    winsize_cam = 450
else:
    winsize_cam = 300
#winsize_cam = 525

window = np.s_[int(x_camcent-winsize_cam):int(x_camcent+winsize_cam),int(y_camcent-winsize_cam):int(y_camcent+winsize_cam)]
fig, ax = plt.subplots(2,2,figsize=(16,16))
ax[0,0].imshow(im_before_pb[window], cmap='flag')
ax[0,1].imshow(io.imread(mt_imgfiles[num_pb])[window], cmap='flag')
ax[1,0].imshow(im_pb1[window])
ax[1,1].imshow(im_pb2[window])

for a in ax:
    for _a in a:
        _a.set_xticklabels([])
        _a.set_yticklabels([])

plt.show()
#%%
# Temporal cross correlation
test_file = io.imread(mt_trimmed[0])
steps = 10
use_mask = False
dict_color = dict({-1:'green',0:'rebeccapurple',1:'tomato',2:'dodgerblue'})

x = np.arange(0,np.shape(test_file)[0],1)
y = np.arange(0,np.shape(test_file)[1],1)
Y,X = np.meshgrid(y,x)

r_min = 0
step = 20

n_correlate = 100
correlation_mt = np.zeros((len(mt_trimmed)-1,n_correlate))
correlation_mot = np.zeros((len(mt_trimmed)-1,n_correlate))
r_intervals = np.zeros((len(mt_trimmed)-1,n_correlate))

v_xmtall = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
v_ymtall = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
v_xmotall = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
v_ymotall = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))

radii = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
angles = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
r_max = np.zeros(len(mt_trimmed) - 1)

#df = pd.DataFrame([])
plt.clf()
for n in tqdm(range(len(mt_trimmed)-1)):
    im1 = io.imread(mt_trimmed[n])
    im1_norm = amp.image_processing.normalize(im1)

    im2 = io.imread(mt_trimmed[n+1])
    im2_norm = amp.image_processing.normalize(im2)

    im_mot1 = io.imread(mot_trimmed[n])
    im_mot2 = io.imread(mot_trimmed[n+1])

    v_y, v_x = registration.optical_flow_tvl1(im1_norm,im2_norm, tightness=0.3, attachment=30, prefilter=True)
    v_ymtall[:,:,n] = v_y - np.average(v_y)
    v_xmtall[:,:,n] = v_x - np.average(v_x)

    im1_mask = amp.image_processing.image_mask(im1,hw=10)
    im_gauss = filters.gaussian(im1, sigma=20) * (2**16-1)

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
    if len(df_regprops) > 0:
        df_max = df_regprops[df_regprops['area']==df_regprops['area'].max()]
        equiv_diam = df_max['equivalent_diameter'].values[0]
        major_axis_length = df_max['major_axis_length'].values[0]
        minor_axis_length = df_max['minor_axis_length'].values[0]

        r_max[n] = np.mean([major_axis_length,minor_axis_length])/2
        x_cent = df_max['centroid-1'].values[0]
        y_cent = df_max['centroid-0'].values[0]

        if np.sqrt((x_cent - winsize_cam)**2 + (y_cent - winsize_cam)**2) > 200:
            x_cent = winsize_cam
            y_cent = winsize_cam
            r_max[n] = 430
    else:
        x_cent = winsize_cam
        y_cent = winsize_cam
        r_max[n]=430
    #y_cent = np.shape(im1)[0]/2+0.01
    #x_cent = np.shape(im1)[1]/2+0.01
    dr = (430 - r_min) / n_correlate

    radii[:,:,n] = np.sqrt((X - x_cent)**2 + (y_cent - Y)**2) * um_per_pxl
    angles[:,:,n] = np.arctan2(y_cent - Y,X - x_cent)

    if n < 30:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].imshow(im1)
        ax[0].quiver(Y[::step,::step], X[::step,::step], v_xmtall[::step,::step,n], v_ymtall[::step,::step,n],
                    color='dodgerblue', units='dots', angles='xy', scale_units='xy')
        ax[1].hist(np.sqrt((v_xmtall[:,:,n])**2 + (v_ymtall[:,:,n]**2)).flatten(), bins=20)
        plt.show()
r_max *= um_per_pxl
#%%
x = np.arange(0,np.shape(test_file)[0],1)
y = np.arange(0,np.shape(test_file)[1],1)
Y,X = np.meshgrid(y,x)

r_min = 0
step = 20

n_correlate = 100
correlation_mt = np.zeros((len(mt_trimmed)-1,n_correlate))
correlation_mot = np.zeros((len(mt_trimmed)-1,n_correlate))
r_intervals = np.zeros((len(mt_trimmed)-1,n_correlate))

v_xmtall = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
v_ymtall = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
im_mask = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))

radii = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
angles = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
r_max = np.zeros(len(mt_trimmed) - 1)

#df = pd.DataFrame([])
plt.clf()
for n in tqdm(range(len(mt_trimmed)-1)):
    im1 = io.imread(mt_trimmed[n])
    im1_norm = amp.image_processing.normalize(im1)

    im2 = io.imread(mt_trimmed[n+1])
    im2_norm = amp.image_processing.normalize(im2)

    v_y, v_x = registration.optical_flow_tvl1(im1_norm,im2_norm, tightness=0.3, attachment=30)
    v_ymtall[:,:,n] = v_y - np.average(v_y)
    v_xmtall[:,:,n] = v_x - np.average(v_x)

    im_mean = (im1 > filters.threshold_mean(im1))*1
    im_closed = morphology.area_closing(im_mean)
    im_mask[:,:,n] = im_closed.copy()
    regprops = measure.regionprops(im_closed, im1)
    for reg in regprops:
        x_cent, y_cent = reg.weighted_centroid
        r_max[n] = (reg.major_axis_length + reg.minor_axis_length)/2 * um_per_pxl
    
    v_xmask = v_xmtall[:,:,n].copy()
    v_xmask[(im_closed==0)] = np.nan

    v_ymask = v_ymtall[:,:,n].copy()
    v_ymask[(im_closed==0)] = np.nan

    radii[:,:,n] = np.sqrt((X - x_cent)**2 + (y_cent - Y)**2) * um_per_pxl
    angles[:,:,n] = np.arctan2(y_cent - Y,X - x_cent)
    mag = np.sqrt(v_xmask**2 + v_ymask**2)
    if n < 0:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].imshow(im1)
        ax[0].quiver(Y[::step,::step], X[::step,::step], v_xmask[::step,::step], v_ymask[::step,::step],
                    color='dodgerblue', units='dots', angles='xy', scale_units='xy')
        ax[1].scatter(angles[:,:,n],mag, color='dodgerblue', alpha=0.01)
        ax[1].set_xlim(-np.pi,np.pi)
        plt.show()
#%%
# Compute strain rates
Dxx = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
Dxy = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))
Dyy = np.zeros((np.shape(test_file)[0],np.shape(test_file)[1],len(mt_trimmed)-1))

for n in tqdm(range(Dxx.shape[2])):
    Dxx[:,:,n], Dxy[:,:,n], Dyy[:,:,n] = compute_strains(v_xmtall[:,:,n],v_ymtall[:,:,n])
    Dxx[:,:,n] = mask_value(Dxx[:,:,n],im_mask[:,:,n], num=True)
    Dxy[:,:,n] = mask_value(Dxy[:,:,n],im_mask[:,:,n], num=True)
    Dyy[:,:,n] = mask_value(Dyy[:,:,n],im_mask[:,:,n], num=True)

for n in range(30):
    fig, ax = plt.subplots(2,2,figsize=(12,12))
    ax[0,0].quiver(X[::step,::step],Y[::step,::step], v_xmtall[::step,::step,n],
                    v_ymtall[::step,::step,n], 
                    units='dots', angles='xy', scale_units='xy', color='dodgerblue')
    ax[0,1].imshow(Dxx[:,:,n], cmap='viridis')
    ax[1,0].imshow(Dxy[:,:,n], cmap='viridis')
    ax[1,1].imshow(Dyy[:,:,n], cmap='viridis')
#%%
import numba
@numba.jit(nopython=True)
def autocorrelation_optflo(Ax,Ay,autocorrelation):
    autocorrelation = np.zeros(np.shape(Ax)[2])

    for n in range(np.shape(Ax)[2]):
        for m in range(n,np.shape(Ax)[2]):
            autocorrelation[m-n] += np.sum(Ax[:,:,n] * Ax[:,:,m] + Ay[:,:,n] * Ay[:,:,m])
    return autocorrelation
#%%
mt_xc = np.zeros(np.shape(v_xmtall)[2])
mt_xc = autocorrelation_optflo(np.nan_to_num(v_xmtall), np.nan_to_num(v_ymtall), mt_xc)
mt_xc /= mt_xc[0]
frames = np.arange(0, len(mt_xc), 1) * df_info['time interval (s)'].values[0]
#%%
mt_xc = np.zeros(np.shape(v_xmtall)[2])
mot_xc = np.zeros(np.shape(v_xmtall)[2])
# Plot cross correlation by time. Set first frame as reference time
for n in tqdm(range(np.shape(v_xmtall)[2])):
    for m in range(n,np.shape(v_xmtall)[2]):
        mt_xc[m-n] += np.sum(v_xmtall[:,:,n] * v_xmtall[:,:,m] + v_ymtall[:,:,n] * v_ymtall[:,:,m])
        mot_xc[m-n] += np.sum(v_xmotall[:,:,n] * v_xmotall[:,:,m] + v_ymotall[:,:,n] * v_ymotall[:,:,m])
mt_xc /= mt_xc[0]
mot_xc /= mot_xc[0]
frames = np.arange(0, len(mt_xc), 1) * df_info['time interval (s)'].values[0]

mt_smooth = filters.gaussian(mt_xc,sigma=1)
mot_smooth = filters.gaussian(mot_xc,sigma=1)

#%%
for n in range(len(mt_trimmed)-1):
    r_intervals[n,:], correlation_mt[n,:] = amp.stats.correlation_optflo(radii[:,:,n], v_xmtall[:,:,n], v_ymtall[:,:,n],
                                                                        r_min=0, r_max=r_max[n], n_positions=n_correlate,
                                                                        dr = r_max[n]/n_correlate)
#%%
for n in range(10):
    plt.figure(figsize=(10,10))

    plt.scatter(r_intervals[n,:],correlation_mt[n,:],label='%i' %n, color='dodgerblue', alpha=(10-n)/10)
    plt.legend()
#%%
nr, nc = im1.shape
# build an RGB image with the unregistered sequence
seq_im = np.zeros((nr, nc, 3))
seq_im[:,:,0] = (amp.image_processing.normalize(im2))
seq_im[:,:,1] = (amp.image_processing.normalize(im1))
seq_im[:,:,2] = (amp.image_processing.normalize(im1))

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.imshow(seq_im)
#%%
mt_smooth = filters.gaussian(mt_xc, sigma=1)
fig, ax = plt.subplots(1,2,figsize=(16,8))
ax[0].scatter(frames,mt_xc,color='dodgerblue',s=16)
ax[0].plot(frames, mt_smooth, color='dodgerblue', lw=2, alpha=0.5)
ax[1].scatter(frames,mot_xc,color='tomato',s=16)
ax[1].plot(frames, mot_smooth, color='tomato', lw=2, alpha=0.5)

for a in ax:
    a.set_xlabel('lag time [sec]', fontsize=16)
    a.set_xlim(frames[0]-5,frames[-1]+5)
    a.set_ylim(-0.01,1.01)

ax[0].set_ylabel('normalized autocorrelation', fontsize=16)

ax[0].set_title('microtubules', fontsize=20)
ax[1].set_title('motors', fontsize=20)

fig.tight_layout()
plt.show()
#plt.savefig('../figures/FigX_autocorrelation_time_MT_motors.pdf',
#            bbox_inches='tight')
# %%
n_vals = [1, 10, 40]
fig, ax = plt.subplots(3,2,figsize=(20,30))
step=10
count = 0
win = np.s_[125:375,125:375]
for n in n_vals:
    im1 = io.imread(mt_trimmed[n])
    im2 = io.imread(mt_trimmed[n+1])

    x = np.arange(0,im1.shape[0],1)
    y = np.arange(0,im1.shape[1],1)

    Y,X = np.meshgrid(y,x)

    im1_norm = amp.image_processing.normalize(im1)
    im2_norm = amp.image_processing.normalize(im2)

    nr, nc = im1.shape
    im_seq = np.zeros((nr,nc,3))

    im_seq[:,:,0] = im2_norm
    im_seq[:,:,1] = im1_norm
    im_seq[:,:,2] = im1_norm

    v_y, v_x = registration.optical_flow_tvl1(im1_norm, im2_norm,
                                            tightness=0.3, attachment=30)
    v_yavg = v_y - np.mean(v_y)
    v_xavg = v_x - np.mean(v_x)

    ax[count,0].imshow(np.uint8(im1_norm*255))
    ax[count,0].quiver(Y[::step,::step], X[::step,::step], v_xavg[::step,::step], v_yavg[::step,::step],
            color='dodgerblue', units='xy', angles='xy', scale_units='xy')

    ax[count,1].imshow(im_seq)
    count += 1
for a in ax.flatten():
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
fig.tight_layout()
# %%
for n in range(20):
    im = io.imread(mt_trimmed[n])
    im_segment = morphology.area_closing(im > filters.threshold_mean(im))*1
    regprops = measure.regionprops(im_segment, intensity_image=im)
    for reg in regprops:
        x_cent, y_cent = reg.weighted_centroid

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(im * morphology.area_closing(im > filters.threshold_mean(im)))
    ax[0].scatter(x_cent, y_cent)
    ax[1].imshow(im, cmap='flag')
# %%
for n in range(20):
    im = io.imread(mt_trimmed[n])
    im_segment = morphology.area_closing(im > filters.threshold_mean(im))*1
    regprops = measure.regionprops(im_segment, intensity_image=im)
    for reg in regprops:
        x_cent, y_cent = reg.weighted_centroid

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(im * morphology.area_closing(im > filters.threshold_mean(im)))
    ax[0].scatter(x_cent, y_cent)
    ax[1].imshow(im, cmap='flag')
#%%
im1 = io.imread(mt_trimmed[0])
im1_norm = amp.image_processing.normalize(im1)

im2 = io.imread(mt_trimmed[1])
im2_norm = amp.image_processing.normalize(im2)

im_mean = morphology.area_closing(im1 > filters.threshold_mean(im1))*1
im_otsu = morphology.area_closing(im1 > filters.threshold_otsu(im1))*1
im_pbreg = im_mean - im_otsu

v_y, v_x = registration.optical_flow_tvl1(im1, im2, tightness=0.3, attachment=30)
v_yavg = v_y - np.average(v_y)
v_yavg[(im_pbreg==0)] = np.nan

v_xavg = v_x - np.average(v_x)
v_xavg[(im_pbreg==0)] = np.nan

x = np.arange(0, im1.shape[0], 1)
y = np.arange(0, im1.shape[1], 1)
Y, X = np.meshgrid(y,x)
#%%
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].imshow(im_pbreg * im1)
ax[0].quiver(Y[::step,::step], X[::step,::step], v_xavg[::step,::step], v_yavg[::step,::step],
            color='dodgerblue', angles='xy', units='dots', scale_units='xy')
ax[1].hist(np.sqrt(v_xavg**2 + v_yavg**2).flatten(), bins=20)
# %%
