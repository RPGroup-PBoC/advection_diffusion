#%%
import sys
sys.path.insert(0,'../')
import active_matter_pkg as amp
from multiprocess import Pool, cpu_count
from tqdm import tqdm
amp.viz.plotting_style()

filelist = '../../analyzed_data/analyzing_filenames.txt'

with open(filelist, 'r') as filestream:
    lines = [line[:-1] for line in filestream if 'slide' in line]
    lines[-1] += '1'

def process_cropping(file):
    largefield = ['11-17-2022_slide1_lane2_pos5_k401bac_iLidmicro_longMT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip0_photobleach_frame37_1',
                '11-17-2022_slide1_lane2_pos6_k401bac_iLidmicro_longMT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip0_photobleach_frame43_1',
                '11-30-2022_slide1_lane3_pos5_ncd236_iLidmicro_longMT647_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_nophotobleach_frame16_1',
                '02-06-2023_slide1_lane6_pos2_ncd236_iLidmicro_MT647_750uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame12_photobleach_1',
                '02-06-2023_slide1_lane6_pos1_ncd236_iLidmicro_MT647_750uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame10_photobleach_1']

    data_root, _ = amp.io.identify_root(file)

    _, _, mt_trimmed, _, _ = amp.io.tiff_walk(data_root, parse_channels=True)

    if (len(mt_trimmed) > 3):
        return

    if file in largefield:
        amp.image_processing.simple_crop(data_root)

    else:
        try:
            amp.image_processing.crop_imageset(data_root, show_process=False)
        except:
            print(file, ' unable to crop. Using simple crop.')
            amp.image_processing.simple_crop(data_root)
    return

def apply_parallel(filelist, func):
    with Pool(cpu_count()) as p:
        ret_list = list(tqdm(p.imap(func, filelist), total=len(filelist)))

    return ret_list

#%%
if __name__ == '__main__':
    df_compiled = apply_parallel(lines, process_cropping)

# %%
lines = [file for file in lines if '02-02-2023' in file]
for file in lines:
    data_root, _ = amp.io.identify_root(file)
    amp.image_processing.crop_imageset(data_root)
# %%
# For checking files
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
import os
lines = [file for file in lines if ('02-06-2023' in file) or ('02-07-2023' in file)]

for file in lines:
    data_root, _ = amp.io.identify_root(file)
    _, _, _, _, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)
    if len(subdirectory)>0:
        if any('.csv' in filename for filename in os.listdir(data_root)):
            df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
            data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    mt_imgfiles, _, _, _, _ = amp.io.tiff_walk(data_root, parse_channels=True)

    df_info = amp.io.parse_filename(data_root)

    num_pb = df_info['photobleach frame number'].values[0]

    fig, ax = plt.subplots(1,2,figsize=(16,8))
    ax[0].imshow(io.imread(mt_imgfiles[num_pb-2]))
    ax[1].imshow(io.imread(mt_imgfiles[num_pb-1]))
    ax[1].set_title(num_pb)
    ax[0].set_title(file)
    plt.show()
#%%
lines = ['02-25-2023_slide1_lane3_pos1_ncd236_iLidmicro_MT647_75uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame22_photobleach_1',
         '02-25-2023_slide1_lane3_pos2_ncd236_iLidmicro_MT647_75uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame19_photobleach_1',
         '02-25-2023_slide1_lane3_pos3_ncd236_iLidmicro_MT647_75uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-25-2023_slide1_lane3_pos4_ncd236_iLidmicro_MT647_75uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-25-2023_slide1_lane3_pos5_ncd236_iLidmicro_MT647_75uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame12_photobleach_1',
         '02-25-2023_slide1_lane4_pos1_ncd236_iLidmicro_MT647_75uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-25-2023_slide1_lane5_pos3_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame18_photobleach_1',
         '02-25-2023_slide1_lane5_pos4_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame19_photobleach_1',
         '02-25-2023_slide1_lane5_pos5_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame17_photobleach_1',
         '02-25-2023_slide1_lane5_pos6_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame18_photobleach_1',
         '02-25-2023_slide1_lane5_pos6_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame22_photobleach_1',
         '02-25-2023_slide1_lane5_pos7_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame20_photobleach_1',
         '02-25-2023_slide1_lane5_pos8_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame17_photobleach_1',
         '02-25-2023_slide1_lane5_pos8_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame19_photobleach_1',
         '02-25-2023_slide1_lane5_pos10_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame15_photobleach_1',
         '02-25-2023_slide1_lane6_pos1_ncd236_iLidmicro_MT647_75uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame19_photobleach_1',
         '02-25-2023_slide1_lane6_pos2_ncd236_iLidmicro_MT647_75uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame18_photobleach_1',
         '02-22-2023_slide1_lane1_pos3_k401bac_iLidmicro_MT647_10xpluronic_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame9_photobleach_longrun_1',
         '02-22-2023_slide1_lane1_pos4_k401bac_iLidmicro_MT647_10xpluronic_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame11_photobleach_longrun_1',
         '02-22-2023_slide1_lane1_pos5_k401bac_iLidmicro_MT647_10xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame18_photobleach_1',
         '02-22-2023_slide1_lane1_pos6_k401bac_iLidmicro_MT647_10xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame17_photobleach_1',
         '02-22-2023_slide1_lane1_pos7_k401bac_iLidmicro_MT647_10xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame15_photobleach_1',
         '02-22-2023_slide1_lane1_pos9_k401bac_iLidmicro_MT647_10xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame23_photobleach_1',
         '02-22-2023_slide1_lane1_pos10_k401bac_iLidmicro_MT647_10xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame17_photobleach_1',
         '02-23-2023_slide1_lane2_pos4_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame28_photobleach_1',
         '02-23-2023_slide1_lane2_pos5_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame28_photobleach_1',
         '02-23-2023_slide1_lane2_pos5_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1',
         '02-23-2023_slide1_lane2_pos5_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame32_photobleach_2',
         '02-23-2023_slide1_lane2_pos6_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame26_photobleach_1',
         '02-23-2023_slide1_lane2_pos7_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame28_photobleach_1',
         '02-23-2023_slide1_lane2_pos8_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1',
         '02-23-2023_slide1_lane2_pos9_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame24_photobleach_1',
         '02-23-2023_slide1_lane5_pos2_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-23-2023_slide1_lane5_pos3_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame24_photobleach_1',
         '02-23-2023_slide1_lane5_pos4_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame26_photobleach_1',
         '02-23-2023_slide1_lane5_pos5_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame21_photobleach_1',
         '02-23-2023_slide1_lane5_pos5_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame21_photobleach_2',
         '02-23-2023_slide1_lane5_pos6_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame18_photobleach_1',
         '02-23-2023_slide1_lane5_pos7_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame19_photobleach_1',
         '02-23-2023_slide1_lane5_pos8_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame25_photobleach_1',
         '02-23-2023_slide1_lane5_pos9_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame23_photobleach_1',
         '02-23-2023_slide1_lane5_pos10_k401bac_iLidmicro_MT647_3xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-23-2023_slide1_lane6_pos1_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame28_photobleach_1',
         '02-23-2023_slide1_lane6_pos2_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1',
         '02-23-2023_slide1_lane6_pos3_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame29_photobleach_1',
         '02-23-2023_slide1_lane6_pos4_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame34_photobleach_1',
         '02-23-2023_slide1_lane6_pos5_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame25_photobleach_1',
         '02-23-2023_slide1_lane6_pos6_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame27_photobleach_1',
         '02-23-2023_slide1_lane6_pos7_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame29_photobleach_1',
         '02-23-2023_slide1_lane6_pos8_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame24_photobleach_1',
         '02-23-2023_slide1_lane6_pos9_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame23_photobleach_1',
         '02-23-2023_slide1_lane6_pos10_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame28_photobleach_1',
         '02-23-2023_slide1_lane7_pos3_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame16_photobleach_1',
         '02-24-2023_slide1_lane1_pos2_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame18_photobleach_1',
         '02-24-2023_slide1_lane1_pos3_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-24-2023_slide1_lane1_pos4_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame22_photobleach_1',
         '02-24-2023_slide1_lane1_pos5_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame21_photobleach_1',
         '02-24-2023_slide1_lane1_pos6_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-24-2023_slide1_lane1_pos7_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-24-2023_slide1_lane1_pos8_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1',
         '02-24-2023_slide1_lane1_pos9_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame18_photobleach_1',
         '02-24-2023_slide1_lane1_pos10_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame18_photobleach_1',
         '02-24-2023_slide1_lane1_pos10_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame18_photobleach_1',
         '02-24-2023_slide1_lane1_pos11_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame19_photobleach_1',
         '02-24-2023_slide1_lane1_pos12_k401bac_iLidmicro_MT647_0p1xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame25_photobleach_1',
         '02-16-2023_slide1_lane7_pos4_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame8_photobleach_1',
         '02-16-2023_slide1_lane7_pos6_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame10_photobleach_1',
         '02-16-2023_slide1_lane7_pos7_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame11_photobleach_1',
         '02-16-2023_slide1_lane7_pos8_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame10_photobleach_1',
         '02-16-2023_slide1_lane7_pos9_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame17_photobleach_1',
         '02-16-2023_slide1_lane8_pos2_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame13_photobleach_1',
         '02-16-2023_slide1_lane8_pos2_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame14_photobleach_1',
         '02-16-2023_slide1_lane8_pos3_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame16_photobleach_1',
         '02-16-2023_slide1_lane8_pos4_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame16_photobleach_1',
         '02-16-2023_slide1_lane8_pos5_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame17_photobleach_1',
         '02-16-2023_slide1_lane8_pos6_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame18_photobleach_1',
         '02-16-2023_slide1_lane8_pos7_k401bac_iLidmicro_MT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame15_photobleach_1']

if __name__ == '__main__':
    df_compiled = apply_parallel(lines, process_cropping)
# %%
list2 = ['02-23-2023_slide1_lane2_pos6_k401bac_iLidmicro_MT647_0xpluronic_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame26_photobleach_1',
         '02-25-2023_slide1_lane5_pos10_k401bac_iLidmicro_MT647_500uM_ATP_3s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame15_photobleach_1']

for file in list2:
    data_root, _ = amp.io.identify_root(file)
    amp.image_processing.simple_crop(data_root)
# %%
import matplotlib.pyplot as plt
from skimage import io

for file in lines:
    data_root, _ = amp.io.identify_root(file)
    _, _, test, _, _ = amp.io.tiff_walk(data_root, parse_channels=True)
    plt.imshow(io.imread(test[0]))
    plt.title(file)
    plt.show()
# %%
