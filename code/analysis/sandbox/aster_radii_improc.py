#%%
import numpy as np
import pandas as pd
import os
from skimage import io, morphology, feature, filters, img_as_float, segmentation, measure, draw, color, transform
import matplotlib.pyplot as plt
import active_matter_pkg as amp
amp.viz.plotting_style()

df_graticule = pd.read_csv('../analyzed_data/objective_pxl_micron_scale.csv', sep=',')

activation_root = '../microscope_files/laser_150um_circle_activation_fullfield.tif'
template = io.imread(activation_root)[:,:,1]

data_root = '../../../data/active_stress/210201_ncd/210201_slide1_lane3_pos3_ncd_20s_intervals_larger_activation_10ms_DLPBlue_200ms_DLPRed_100ms_DLPYellow_frame20_lasergridline_1'
fileset = amp.io.find_all_tiffs(data_root)

if 'frame' in data_root:
    num_frame = data_root.find('frame')
    num_uscore = num_frame + data_root[num_frame:].find('_')
    # Photobleaching occurs prior to the activation cycle listed as frame## in the filename
    # Then there is indexing by 0 in python, thus subtracting by 2
    num_pb = int(data_root[num_frame+5:num_uscore]) - 2

mt_imgfiles = np.sort([imfile for imfile in fileset if 'DLP_Red_000.tif' in imfile])
mot_imgfiles = np.sort([imfile for imfile in fileset if 'DLP_Yellow_000.tif' in imfile])

im_before_pb = io.imread(mt_imgfiles[num_pb-1])
im_pb1 = io.imread(mt_imgfiles[num_pb+1])
im_pb2 = io.imread(mt_imgfiles[num_pb+2])

blur_template = filters.gaussian(im_before_pb, sigma=100)
#im_subt = im_before_pb - blur_template * (2**16 - 1)
im_subt = im_before_pb - io.imread(mt_imgfiles[0])
im_subt[im_subt < 0] = 0
thresh = filters.threshold_li(im_subt)
template = (im_subt < thresh) * (2**16 - 1)

seed = np.copy(template)
seed[1:-1,1:-1] = template.max()
filled = morphology.reconstruction(seed, template, method='erosion')

#%%
area_thresh = 10000

filled = morphology.remove_small_objects(filled > 0, area_thresh)
im_label = measure.label(filled)
regionprops = measure.regionprops(im_label, im_before_pb)

area = np.zeros(len(regionprops))
for n in range(1,im_label.max()+1):
    if regionprops[n-1].area < area_thresh:
        im_label[im_label==n] = 0

ind_remainder = np.unique(im_label)[1:]
y_cent, x_cent = np.shape(im_before_pb)[0]/2, np.shape(im_before_pb)[1]/2
find_center = np.zeros((2, len(ind_remainder)))
find_center[0,:] = ind_remainder
for n in range(len(ind_remainder)):
    y_dist = regionprops[ind_remainder[n]-1].centroid[0] - y_cent
    x_dist = regionprops[ind_remainder[n]-1].centroid[1] - x_cent

    find_center[1,n] = np.sqrt(x_dist**2 + y_dist**2)

# Find minimum distance from center. Choose as index representing activation
# region
ind_mindist = ind_remainder[find_center[1,:]==find_center[1,:].min()]
im_label[im_label!=ind_mindist] = 0

im_postproc = im_before_pb * im_label

hough_radii = np.arange(300,550,10)
hough_res = transform.hough_circle(im_label, hough_radii)

# Select most prominent circle
accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, 
                                                        total_num_peaks=1)

image = color.gray2rgb(im_label)

for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = draw.circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
    image[circy, circx] = (220,20,20)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.imshow(im_label)
ax.imshow(image)

df = pd.DataFrame([[center_y, center_x, radius]],
                columns=['center_y', 'center_x', 'radius'])
df['filename'] = mt_imgfiles[num_pb-1]

df.to_csv('../analyzed_data/hough_circle_aster_contraction.csv', sep=',')
# %%
