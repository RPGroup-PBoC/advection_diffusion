#%%
# Macroscopic deformations of fluorescent unit cells
import sys
sys.path.insert(0, '../')
import active_matter_pkg as amp
import pandas as pd
from tqdm import tqdm
from multiprocess import Pool, cpu_count
tqdm.pandas()
amp.viz.plotting_style()

def analyze_photobleach_single(file):
    intensity_thresh = 0.995
    n_start = 0
    n_tot = 20
    small_area = 800
    large_area = 4000

    # Include camera offset
    offset = 1920
    
    # Include threshold to number of layers added to preserve total intensity
    layer_thresh = 50

    return amp.image_processing.analyze_photobleach(file, offset, n_tot, 
                                                    intensity_thresh, layer_thresh, 
                                                    n_start=n_start, new_thresholding=True,
                                                    small_area = small_area, large_area=large_area,
                                                    thresh_method='custom')

def apply_parallel(filelist, func):
    with Pool(cpu_count()) as p:
        ret_list = list(tqdm(p.imap(func, filelist), total=len(filelist)))

    return pd.concat(ret_list)

def func_group_apply(df, param_list):
    return df.groupby(param_list).apply(analyze_photobleach_single)

filelist = '../../analyzed_data/analyzing_filenames.txt'

with open(filelist, 'r') as filestream:
    files = [line[:-1] for line in filestream if 'slide' in line]
#%%
files = [file for file in files if ('02-06-2023' in file) or ('02-07-2023' in file)]

if __name__ == '__main__':
    df_compiled = apply_parallel(files, analyze_photobleach_single)

# Remove merged data
df_compiled = df_compiled[df_compiled['merged']==0]

df_full = pd.read_csv('../../analyzed_data/unitcell_features_allmotors_multithreaded.csv', sep=',')

df_compiled = pd.concat([df_full, df_compiled], ignore_index=True)

df_compiled.to_csv('../../analyzed_data/unitcell_features_allmotors_multithreaded.csv', sep=',')

# %%
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
    df_compiled = apply_parallel(lines, analyze_photobleach_single)
#%%
df_comp = pd.read_csv('../../analyzed_data/unitcell_features_allmotors_multithreaded.csv', sep=',')
df_comp = pd.concat([df_comp, df_compiled], ignore_index=True)
df_comp = df_comp.loc[:, ~df_comp.columns.str.contains('Unnamed')]
df_comp.to_csv('../../analyzed_data/unitcell_features_allmotors_multithreaded.csv', sep=',')
# %%