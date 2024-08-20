#%%
# Macroscopic deformations of fluorescent unit cells
import sys
sys.path.insert(0, '../')
import active_matter_pkg as amp
import os
from skimage import io, filters, feature, morphology, measure, segmentation
from scipy import ndimage as ndi
from scipy import spatial, optimize
from scipy.stats import linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
amp.viz.plotting_style()

#root = '../../../data/active_stress/k401_photobleach'
root = '../../../data/active_stress/photobleach_data//'
data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])

#data_root = str([directory for directory in data_directory if '210519_slide2_lane2_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1' in directory][0])
data_root = str([directory for directory in data_directory if '210518_slide1_lane3_pos3_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_50ms_DLPBlue_skip1_frame40_photobleach_1' in directory][0])
#data_root = str([directory for directory in data_directory if '220425_slide1_lane2_pos2_10s_intervals_ncd236_microiLid_10ms_AllOff_100ms_DLPRed_50ms_DLPBlue_skip1_8thcircle_frame20_photobleach_1' in directory][0])
#data_root = str([directory for directory in data_directory if '210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1' in directory][0])
#data_root = str([directory for directory in data_directory if '210426_slide1_lane2_pos1' in directory][0])

mt_imgfiles, mot_imgfiles, mt_trimmed, mot_trimmed, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

if len(subdirectory)>0:
    if any('filename_order.csv' in filename for filename in os.listdir(data_root)):
        df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
        data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    if any('image_crop.csv' in filename for filename in os.listdir(data_root)):
        df_crop = pd.read_csv(os.path.join(data_root,'image_crop.csv'), sep=',')
df_info = amp.io.parse_filename(data_root)
df_graticule = pd.read_csv('../../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
um_per_pxl = df_graticule['micron_per_pixel'].values[0]
activation_radius = 125 #µm
df_t2c = pd.read_csv('../../analyzed_data/time_to_contraction.csv', sep=',')
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
# %%
df_contractionspeed = pd.DataFrame([])
filelist = ['210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1',
            '210421_slide1_lane3_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_200ms_DLPYellow_200ms_DLPRed_50ms_DLPBlue_skip1_frame5_photobleach_1',
            '210423_slide1_lane2_pos1','210426_slide1_lane1_pos1',
            '210426_slide1_lane1_pos2','210426_slide1_lane2_pos1',
            '210518_slide1_lane3_pos3_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_50ms_DLPBlue_skip1_frame40_photobleach_1',
            '210519_slide2_lane2_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1',
            '210520_slide1_lane3_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame20_photobleach_1',
            '220425_slide1_lane2_pos2_10s_intervals_ncd236_microiLid_10ms_AllOff_100ms_DLPRed_50ms_DLPBlue_skip1_8thcircle_frame20_photobleach_1',
            '220425_slide1_lane2_pos4_10s_intervals_ncd236_microiLid_10ms_AllOff_100ms_DLPRed_50ms_DLPBlue_skip1_8thcircle_frame26_photobleach_1',
            '220425_slide1_lane2_pos5_10s_intervals_ncd236_microiLid_10ms_AllOff_100ms_DLPRed_50ms_DLPBlue_skip1_8thcircle_frame28_photobleach_1',
            '220425_slide1_lane2_pos6_10s_intervals_ncd236_microiLid_10ms_AllOff_100ms_DLPRed_50ms_DLPBlue_skip1_8thcircle_frame30_photobleach_1']

root_start = '../../../data/active_stress/'
root_ncd = 'photobleach_data'

numpb_dict = dict({'210426_slide1_lane1_pos1':34,
            '210426_slide1_lane1_pos2':31,
            '210426_slide1_lane2_pos1':87})

n_min = 0
n_max = 20
intensity_thresh = 0.99
