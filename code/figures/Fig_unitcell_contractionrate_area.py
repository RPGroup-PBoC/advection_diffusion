#%%
# Compiling figure from data analysis of unit cells
import os
import sys
sys.path.insert(0,'../')
import active_matter_pkg as amp
from skimage import io
from scipy.stats import linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
tqdm.pandas()
amp.viz.plotting_style()

root = '../../../data/active_stress/photobleach_data'
data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])

df_graticule = pd.read_csv('../../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
um_per_pxl = df_graticule['micron_per_pixel'].values[0]
activation_radius = 125 #µm
df_t2c = pd.read_csv('../../analyzed_data/time_to_contraction.csv', sep=',')

df_compiled = pd.read_csv('../../analyzed_data/unitcell_features_compiled.csv', sep=',')
df_truncated = df_compiled[(~df_compiled['filename'].str.contains('12-05')
                        & (~df_compiled['filename'].str.contains('11-30')))]

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_multithreading.csv', sep=',')
df_compilerates = df_compilerates[df_compilerates['motor dilution']==1.0]

df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
df_contractionspeed = df_contractionspeed[df_contractionspeed['motor dilution']==1.0]

df_info = amp.io.parse_filename(os.path.join(root, '210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1'))

#%%
data_root = str([directory for directory in data_directory if '210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1' in directory][0])
#data_root = str([directory for directory in data_directory if '12-06-2022_slide1_lane1_pos11_ncd236_iLidmicro_longMT647_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame20_photobleach_1' in directory][0])
#data_root = str([directory for directory in data_directory if '01-23-2023_slide2_lane3_pos3_ncd236_iLidmicro_MT647_250uM_ATP_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip1_frame32_photobleach_1_1' in directory][0])

mt_imgfiles, mot_imgfiles, mt_trimmed, mot_trimmed, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

if len(subdirectory)>0:
    if any('filename_order.csv' in filename for filename in os.listdir(data_root)):
        df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
        data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    if any('image_crop.csv' in filename for filename in os.listdir(data_root)):
        df_crop = pd.read_csv(os.path.join(data_root,'image_crop.csv'), sep=',')
df_info = amp.io.parse_filename(os.path.join(root, '210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1'))
atp = 1400
motor = 'ncd236'

df = df_compiled[df_compiled['filename']==data_root]

#num_array = np.array([7,8,11,9,10,16,15,17,18,19,14,13,25,23,24,28,26,
#                   27,22,20,36,29,33,32,34,31,35,30,38,39,40,43,45,44,
#                    42,41,47,51,49,48,50,53,52,46,55,57,60,59,61,56,58])
num_array = np.array([2,4,5,3,10,7,9,11,8,12,13,16,14,15,20,18,22,19,21,24,25])
#num_array = np.arange(1, len(df['cellID'].unique()) + 1, 1)
map_list = np.arange(1,len(num_array)+1,1)

number_mapping = dict({a:b for a,b in zip(num_array,map_list)})

r = np.linspace(0,df_compiled[(df_compiled['motor']==motor) & 
                                (df_compiled['pluronic']==1) &
                                (df_compiled['ATP (uM)']==atp)]['radius'].max()*1.01,100)
t = np.linspace(0,90,1000)
dt = df_info['time interval (s)'].values[0]
corner_pad = 20

_,_,mt_trimmed,_,_ = amp.io.tiff_walk(data_root)
im = io.imread(mt_trimmed[0])
#Place scale bar in im2
im[-corner_pad-10:-corner_pad,int(-corner_pad-50/um_per_pxl):-corner_pad] = im.max()

contraction_median = df_compilerates[(df_compilerates['motor']==motor)
                                    & (df_compilerates['motor dilution']==1.0)
                                    & (df_compilerates['ATP (uM)']==atp)
                                    & (df_compilerates['pluronic']==1)]['mean_rate'].values[0]
area_contraction = (1 - contraction_median * t)**2

df_speedtrunc = df_contractionspeed[(df_contractionspeed['motor']==motor)
                & (df_contractionspeed['ATP (uM)']==atp)
                & (df_contractionspeed['motor dilution']==1.0)
                & (df_contractionspeed['pluronic']==1)]

df_compile = df_compilerates[(df_compilerates['motor']==motor)
                                & (df_compilerates['motor dilution']==1.0)
                                & (df_compilerates['ATP (uM)']==atp)
                                & (df_compilerates['pluronic']==1)]

axis_fontsize = 30
label_fontsize = 36

fig = plt.figure(figsize=(32,8))
ax1 = fig.add_subplot(1,4,1)
ax2 = fig.add_subplot(1,4,2)
ax3 = fig.add_subplot(1,4,3)
ax4 = fig.add_subplot(1,4,4)

ax1.imshow(im)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.text(int(np.shape(im)[0]-corner_pad-25/um_per_pxl), np.shape(im)[0]-corner_pad-20, '50 μm',
            fontsize=30, ha='center', va='bottom', color='white', weight='bold')
ax1.tick_params(axis='both', labelsize=24)
ax2.tick_params(axis='both', labelsize=24)
ax3.tick_params(axis='both', labelsize=24)
ax4.tick_params(axis='both', labelsize=24)

for id,d in df.groupby('cellID'):
    if (len(d) <= 4) or (d['time'].min()>3) or (d['radius'].values[0]>125):
        continue
    ax1.text(d[d['time']==0]['centroid-1']-20, d[d['time']==0]['centroid-0']-20,
                '%i' %number_mapping[id], ha='center', va='center', color='white', fontsize=18)
    d['absolute_time'] = (d['time'] - 1) * dt + 2
    d.loc[d['absolute_time'] < 0, 'absolute_time'] = 0
    ax2.plot(d['absolute_time'], d['radius'],
                color='rebeccapurple', ls='-', lw=2, marker='o')
    if id==24:
        ax2.text(-1.5, d[d['time']==0]['radius']-1.0, '%i' %number_mapping[id], fontsize=14,
                ha='right', va='center')
    elif id==5 or id==26:
        ax2.text(-1.5, d[d['time']==0]['radius']+0.5, '%i' %number_mapping[id], fontsize=14,
                ha='right', va='center')
    elif id==3:
        ax2.text(-1.5, d[d['time']==0]['radius']-0.5, '%i' %number_mapping[id], fontsize=14,
                ha='right', va='center')
    elif id==14:
        ax2.text(-1.5, d[d['time']==0]['radius']+1.0, '%i' %number_mapping[id], fontsize=14,
                ha='right', va='center')
    elif id==10 or id==26:
        ax2.text(-1.5, d[d['time']==0]['radius']+1.5, '%i' %number_mapping[id], fontsize=14,
                ha='right', va='center')
    else:
        ax2.text(-1.5, d[d['time']==0]['radius'], '%i' %number_mapping[id], fontsize=14,
            ha='right', va='center')
ax2.set_ylabel('distance from center of\ncontraction [μm]', fontsize=axis_fontsize)
ax2.set_xlabel('time [s]', fontsize=axis_fontsize)

ax3.scatter(df_speedtrunc[(df_speedtrunc['filename']!=data_root)]['radius'],
            df_speedtrunc[(df_speedtrunc['filename']!=data_root)]['contraction speed'],
            color='gray', s=80, alpha=0.4, label='individual cells')
ax3.scatter(df_speedtrunc[df_speedtrunc['filename']==data_root]['radius'],
            df_speedtrunc[(df_speedtrunc['filename']==data_root)]['contraction speed'], lw=2,
            marker='o',edgecolor='white', facecolor='rebeccapurple', s=100, label='cells from panel B')
contract_rate  = df_compile['mean_rate'].values[0]
ax3.plot(df_compile['distance'],
            df_compile['median'], 
            color='dodgerblue', lw=3, ls='-', label=r'$\alpha$ = %.04f s$^{-1}$' %contract_rate)
ax3.fill_between(df_compile['distance'], 
                    df_compile['cred_low'], 
                    df_compile['cred_high'],
                    color='dodgerblue',alpha=0.3,zorder=-2,label='95$\%$ credible region')
ax3.legend(loc=2, fontsize=20)
ax3.set_ylabel('contraction speed [µm/s]', fontsize=axis_fontsize)
ax3.set_xlabel('radius [µm]', fontsize=axis_fontsize)
ax3.set_xlim(-0.5,120)
for fid,d in df_truncated[(df_truncated['motor']==motor)
                        & (df_truncated['ATP (uM)']==atp)
                        & (df_truncated['pluronic']==1)].groupby(['filename','cellID']):
    f,id = fid
    if ('12-05' in f) or ('11-30' in f):
        continue

    ax4.scatter(d[d['time'] == 0]['time']*dt,
                    d[d['time'] == 0]['area_normalized'], color='gray', s=80,
                    marker='o', alpha=0.05, zorder=-10)
    ax4.scatter((d[(d['time']<10) & (d['time'] > 0)]['time'] - 1) * dt + 2,
                    d[(d['time']<10) & (d['time'] > 0)]['area_normalized'], color='gray',
                    s=80, marker='o', alpha=0.05, zorder=-10)

df_quartiles = pd.DataFrame()
df_ref = df_truncated[(df_truncated['motor']==motor)
                            & (df_truncated['ATP (uM)']==atp)
                            & (df_truncated['pluronic']==1)]
for time,d in df_ref.groupby('time'):
    if time == 0:
        _df_quartiles = pd.DataFrame([[time, np.quantile(d['area_normalized'], 0.25), np.quantile(d['area_normalized'], 0.5), np.quantile(d['area_normalized'], 0.75)]],
                                     columns=('time', '1st quartile', 'median', '3rd quartile'))
    elif time >= 10:
        continue
    else:
        _df_quartiles = pd.DataFrame([[(time - 1) * dt + 2, np.quantile(d['area_normalized'], 0.25), np.quantile(d['area_normalized'], 0.5), np.quantile(d['area_normalized'], 0.75)]],
                                     columns=('time', '1st quartile', 'median', '3rd quartile'))
    df_quartiles = pd.concat([df_quartiles, _df_quartiles], ignore_index = True)


ax4.scatter(np.nan,np.nan,color='gray', marker='o', s=120,
                zorder=-10, label='individual cells')
ax4.scatter(df_quartiles['time'], df_quartiles['median'], edgecolor='tomato', marker='o', s=140,
                facecolor='white', label='median across cells')
ax4.fill_between(df_quartiles['time'], df_quartiles['1st quartile'], df_quartiles['3rd quartile'],
                     color='tomato', alpha=0.5, zorder=0, label='±25% of distribution')
ax4.plot(t, area_contraction,color='dodgerblue',ls='--',lw=4,
             zorder=-4, label='pure contraction bound')

ax4.legend(loc=2, fontsize=20)
ax4.set_ylabel('normalized area', fontsize=axis_fontsize)
ax4.set_xlabel('time [s]', fontsize=axis_fontsize)

#for a in ax.flatten():
#    a.tick_params(axis='both', labelsize=24)
#ax[0,0].set_title(os.path.split(data_root)[-1])
#ax[0].text(-20,0, '(B)', ha='right', va='bottom', fontsize=label_fontsize + 12)
#ax[1].text(-8,73,'(C)', ha='right', va='top', fontsize=label_fontsize + 12)
#x[2].text(-8,0.36,'(D)', ha='right', va='bottom', fontsize=label_fontsize + 12)
#ax[3].text(-10,1.46,'(E)', ha='right', va='bottom', fontsize=label_fontsize + 12)
fig.tight_layout()
fig.savefig('../../figures/FigX_contractionrate_areasize.pdf', bbox_inches='tight',
            facecolor='white')
#%%
# Save just the portion _inside_ the second axis's boundaries
extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('../../figures/FigX_squareimage.pdf', bbox_inches=extent)

# Pad the saved area by 10% in the x-direction and 20% in the y-direction
#fig.savefig('ax2_figure_expanded.png', bbox_inches=extent.expanded(1.1, 1.2))

# %%
extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('../../figures/FigX_unitcellspeed.pdf', bbox_inches=extent2.expanded(1.3,1.3))

extent3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('../../figures/FigX_contractioncompute.pdf', bbox_inches=extent3.expanded(1.4,1.4))

extent4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('../../figures/FigX_normalizedareacompute.pdf', bbox_inches=extent4.expanded(1.3,1.3))
# %%
