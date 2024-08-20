#%%
import os
import active_matter_pkg as amp
from skimage import io, filters, feature, morphology, measure, transform, draw
from scipy import fft, ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
amp.viz.plotting_style()
%matplotlib inline

root = '../../../data/active_stress/photobleach_data'
data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])

data_root = str([directory for directory in data_directory if '210520_slide1_lane2_pos2' in directory][0])
#data_root = str([directory for directory in data_directory if '210518' in directory][0])

mt_imgfiles, mot_imgfiles, mt_trimmed, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

if len(subdirectory)>0:
    if any('filename_order.csv' in filename for filename in os.listdir(data_root)):
        df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
        data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    if any('image_crop.csv' in filename for filename in os.listdir(data_root)):
        df_crop = pd.read_csv(os.path.join(data_root,'image_crop.csv'), sep=',')
df_info = amp.io.parse_filename(data_root)
df_graticule = pd.read_csv('../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
um_per_pxl = df_graticule['micron_per_pixel'].values[0]

#num_pb = 53
num_pb = df_info['photobleach frame number'].values[0]
im_first = io.imread(mt_imgfiles[0])
im_before_pb = io.imread(mt_imgfiles[num_pb-1])
# At this time, the first two images taken after photobleaching are 
# taken with little interval in between. Skipping the first of the two
im_pb1 = io.imread(mt_imgfiles[num_pb+1])
im_pb2 = io.imread(mt_imgfiles[num_pb+2])

# bounding box restricting primarily to activation region
x_camcent = 650
y_camcent = 975
if num_pb < 60:
    winsize_cam = 450
else:
    winsize_cam = 300

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
# %%
def image_mask(image, sigma=30, hw=8, threshold_method='mean', min_size=64):
    im_bk = filters.gaussian(image, sigma=sigma) * (2**16-1)
    im_subt = image - im_bk
    im_subt[im_subt<0] = 0

    if threshold_method=='mean':
        thresh = filters.threshold_mean(im_subt)
    elif threshold_method=='otsu':
        thresh = filters.threshold_otsu(im_subt)
    elif threshold_method=='isodata':
        thresh = filters.threshold_isodata(im_subt)
    else:
        thresh = filters.threshold_mean(im_subt)
    im_thresh = (im_subt > thresh)

    im_binary = morphology.remove_small_objects(im_thresh, min_size=min_size)
    im_label, n_label = measure.label(im_binary, return_num=True)
    im_border = amp.image_processing.border_clear(im_label,edge=hw)
    return (im_border>0)

def find_local_peaks(radon, neighborhood_size=5, threshold=5000):

    data_max = ndimage.filters.maximum_filter(radon, neighborhood_size)
    maxima = (radon == data_max)
    data_min = ndimage.filters.minimum_filter(radon, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y, peaks = [], [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)

        peaks.append(radon[int(y_center),int(x_center)])

    return x,y,np.array(peaks)

def find_radon_angle(im,sigma=0.5,offset_val=1914, image_gradient='None', second=False):
    offset = np.ones(np.shape(im))*offset_val
    im_removeoffset = im - offset

    if image_gradient=='sobel':
        im_sobel = filters.sobel(im_removeoffset)
        im_ = amp.image_processing.normalize(im_sobel) * (2**16-1)
    elif image_gradient=='roberts':
        im_roberts = filters.roberts(im_removeoffset)
        im_ = amp.image_processing.normalize(im_roberts) * (2**16-1)
    elif image_gradient=='None':
        im_ = im_removeoffset.copy()
    im_gauss = filters.gaussian(im_, sigma=sigma)

    radon = transform.radon(im_gauss)

    if len(np.where(radon==radon.max())[0]) != 1:
        return np.nan
    else:
        x,_,peaks = find_local_peaks(radon, neighborhood_size=5, threshold=5000)
        if not second:
            idx = np.where(peaks==np.max(peaks))[0][0]
            return x[idx] * np.pi/180
        else:
            peaks[peaks==np.max(peaks)] = 0
            peaks[peaks==np.max(peaks)] = 0
            idx = np.where(peaks==np.max(peaks))[0][0]
            return x[idx] * np.pi/180

def find_director_field(image, hw=8, start=60, end=460, step=6, sigma=30, apply_mask=True, threshold_method='mean', image_gradient='None',camera_offset=1914):
    x_im = np.arange(start,end,step=step)
    y_im = np.arange(start,end,step=step)

    Y_im, X_im = np.meshgrid(y_im, x_im)
    directors = np.zeros(np.shape(X_im))
    im_mask = image_mask(image, sigma=sigma, hw=hw, threshold_method=threshold_method, min_size=int(hw**2))

    for n_x in range(len(x_im)):
        for n_y in range(len(y_im)):
            x_cent = X_im[n_y,n_x]
            y_cent = Y_im[n_y,n_x]
            x_low, x_high = int(x_cent-hw),int(x_cent+hw)
            y_low, y_high = int(y_cent-hw),int(y_cent+hw)
            window = np.s_[y_low:y_high,x_low:x_high]
            if apply_mask:
                if np.sum(im_mask[window]) >= 3.5*hw**2:
                    directors[n_y,n_x] = find_radon_angle(image[window],offset_val=camera_offset,image_gradient=image_gradient)
                elif (np.sum(im_mask[window]) > 2*hw**2) and (np.sum(im_mask[window]) < 3.5*hw**2):
                    directors[n_y,n_x] = find_radon_angle(image[window],offset_val=camera_offset,image_gradient=image_gradient,second=True)
                else:
                    directors[n_y,n_x] = np.nan
            else:
                directors[n_y,n_x] = find_radon_angle(image[window],offset_val=camera_offset,image_gradient=image_gradient)

    return directors, x_im, y_im

# %%
if 'director_fields' not in os.listdir(data_root):
    os.mkdir(os.path.join(data_root,'director_fields'))

if 'director_fields_csv' not in os.listdir(data_root):
    os.mkdir(os.path.join(data_root,'director_fields_csv'))

for n in range(len(mt_trimmed)):
    im = io.imread(mt_trimmed[n])

    directors, x_im, y_im = find_director_field(im, hw=8, step=8, 
                                                threshold_method='mean',
                                                sigma=30, image_gradient='sobel',
                                                camera_offset=1914, apply_mask=True)
    """
    fig, ax = plt.subplots(1,2,figsize=(16,8))
    ax[0].imshow(im)
    ax[1].imshow(im)

    for n_x in range(len(x_im)):
        for n_y in range(len(y_im)):
            angle = directors[n_y,n_x]
            x_end, y_end = x_im[n_x] - np.cos(angle)*3, y_im[n_y] - np.sin(angle)*3
            x_start, y_start = x_im[n_x] + np.cos(angle)*3, y_im[n_y] + np.sin(angle)*3
            ax[1].plot([y_start,y_end],[x_start,x_end],lw=2.5,color='tomato')
    ax[1].set_xlim([0,np.shape(im)[0]])
    ax[1].set_ylim([np.shape(im)[1],0])

    for a in ax:
        a.set_axis_off()
    fig.tight_layout()
    """
    #plt.savefig(os.path.join(data_dir, subdirectory[1],'director_fields','aster_formation_director_%ith_frame.tif' %n),
    #            bbox_inches='tight',facecolor='white')

    idx = np.where(directors > -1)
    directors_idx = directors[idx[0],idx[1]]
    df = pd.DataFrame(np.array([x_im[idx[0]],y_im[idx[1]],directors[idx[0],idx[1]]]).T,
                        columns=['x','y','angle'])
    df.to_csv(os.path.join(data_root,'director_fields_csv','director_fields_%03d.csv' %n), sep=',')

    #plt.show()
    
    #plt.close(fig)

# %%
# Make a dummy sample consisting of concentric circles
test = np.zeros((300,300))
for n in np.array([110, 90, 70, 55, 45, 25]):
    circ_out = draw.circle(140, 160, n)
    circ_in = draw.circle(140, 160, n-5)
    test[circ_out] += 1
    test[circ_in] -= 1
noise = np.random.normal(scale=0.1, size=np.shape(test))
test_final = filters.gaussian(amp.image_processing.normalize(test + noise), sigma=1)
final = (amp.image_processing.normalize(test_final) * (2**16-1)).astype('uint16')
plt.imshow(final)
# %%
hw=8
im = final.copy()
directors, x_im, y_im = find_director_field(im, hw=hw, start=20, end=280, 
                                            step=4, apply_mask=True, threshold_method='mean',
                                            sigma=30)
plt.figure(figsize=(12,12))
plt.imshow(im)

for n_x in range(len(x_im)):
    for n_y in range(len(y_im)):
        angle = directors[n_y,n_x]
        x_end, y_end = x_im[n_x] - np.cos(angle)*3, y_im[n_y] - np.sin(angle)*3
        x_start, y_start = x_im[n_x] + np.cos(angle)*3, y_im[n_y] + np.sin(angle)*3
        plt.plot([y_start,y_end],[x_start,x_end],lw=2,color='tomato')
plt.xlim([0,np.shape(im)[0]])
plt.ylim([0,np.shape(im)[1]])

ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])

#plt.savefig(os.path.join(data_root,'director_fields','aster_formation_director_%ith_frame.tif' %n),
#            bbox_inches='tight',facecolor='white')

plt.show()
plt.close(fig)

# %%
