#%%
# Compiling figure from data analysis of unit cells
import os
import sys
sys.path.insert(0,'../')
import active_matter_pkg as amp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
amp.viz.plotting_style()

root = '../../../data/active_stress/photobleach_data'
data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])
data_root = str([directory for directory in data_directory if '210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1' in directory][0])

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

df_compiled = pd.read_csv('../../analyzed_data/unitcell_features_compiled.csv', sep=',')
df_truncated = df_compiled[(~df_compiled['filename'].str.contains('12-05')
                        & (~df_compiled['filename'].str.contains('11-30')))]
df_truncated = df_truncated[(df_truncated['ATP (uM)']==1400) & (df_truncated['pluronic']==1) & (df_truncated['motor']=='ncd236')]

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_multithreading.csv', sep=',')
df_compilerates = df_compilerates[(df_compilerates['motor dilution']==1.0)
                                  & (df_compilerates['ATP (uM)']==1400)
                                  & (df_compilerates['pluronic']==1)
                                  & (df_compilerates['motor']=='ncd236')]

df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
df_contractionspeed = df_contractionspeed[(df_contractionspeed['motor dilution']==1.0)
                                          & (df_contractionspeed['ATP (uM)']==1400)
                                          & (df_contractionspeed['pluronic']==1)
                                          & (df_contractionspeed['motor']=='ncd236')]


# %%
axis_fontsize = 9
label_fontsize = 5

fname1 = '2023_slide1_lane1_pos1_ncd236_iLidmicro_MT647_10s_intervals_200ms_DLPRed_50ms_DLPBlue_skip2_frame2_photobleach_1'
fname2 = '210421_slide1_lane3_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_200ms_DLPYellow_200ms_DLPRed_50ms_DLPBlue_skip1_frame5_photobleach_1'

n_cols = 5
n_rows = 6
n_rem = int(2 * np.ceil((len(df_truncated['filename'].unique())) / 5)) - n_rows


n = 0

df_truncated['pb_time'] = (df_truncated['num_pb'] - 1) * df_truncated['time interval (s)'] + 2

df_truncated.loc[df_truncated['filename'].str.contains(fname1),'num_pb'] = df_truncated[df_truncated['filename'].str.contains(fname1)]['pb_time'].values[0] + 32 * 10
df_truncated.loc[df_truncated['filename'].str.contains(fname2),'num_pb'] = 862

fig, ax = plt.subplots(n_rows,n_cols,figsize=(n_cols*2, n_rows*2))

for f,df in df_truncated.sort_values('pb_time').groupby('filename', sort=False):

    c = int(n % 5)
    r = 2 * int(n / 5)

    if r >= n_rows:
        last_filename = f
        break

    df_speedtrunc = df_contractionspeed[(df_contractionspeed['filename']==f)]

    df_compile = df_compilerates

    rad = np.linspace(0, df_truncated['radius'].max()*1.01,100)

    t = np.linspace(0,df_truncated[(df_truncated['time']<10)]['time'].max() * df_truncated['time interval (s)'].values[0],1000)

    #df = df_compiled[(df_compiled['filename']==data_root) & (~df_compiled['filename'].str.contains('-'))]
    
    contraction_median = df_compile['mean_rate'].values[0]
    area_contraction = (1 - contraction_median * t)**2

    ax[r,c].scatter(df_speedtrunc['radius'], df_speedtrunc['contraction speed'],
                color='gray', s=30, alpha=0.4, label='individual unit cells')
    contract_rate  = df_compile['mean_rate'].values[0]
    ax[r,c].plot(df_compile['distance'], df_compile['median'], 
                color='tomato', lw=3, ls='-', label=r'contraction rate = %.04f s$^{-1}$' %contract_rate)
    ax[r,c].fill_between(df_compile['distance'], df_compile['cred_low'], df_compile['cred_high'],
                        color='tomato',alpha=0.3,zorder=-2,label='95$\%$ credible region')
    ax[r,c].legend(loc=2, fontsize=label_fontsize)
    ax[r,c].set_ylabel('contraction speed [µm/s]', fontsize=axis_fontsize)
    ax[r,c].set_xlabel('radius [µm]', fontsize=axis_fontsize)
    ax[r,c].set_ylim(-0.01, 0.51)
    ax[r,c].set_xlim(np.min(rad)-0.01,np.max(rad))

    for id,d in df.groupby('cellID'):

        if d['time'].min() == 0:
            ax[r + 1,c].scatter((d[d['time']>0]['time'] - 1) * d[d['time']>0]['time interval (s)'] + 2,
                            d[d['time']>0]['area_normalized'], color='gray', s=10,
                            marker='o', alpha=0.1)
            ax[r + 1,c].scatter(0, d[d['time']==0]['area_normalized'], color='gray', s=10,
                            marker='o', alpha=0.1)
        else:
            d.loc[:,'time'] = d.loc[:,'time'] - d['time'].min()
            ax[r + 1,c].scatter((d['time']) * d['time interval (s)'],
                            d['area_normalized'], color='gray', s=10,
                            marker='o', alpha=0.1)
    # Compute  the mean area
    for time,d in df.groupby('time'):
        #d = d[~d['filename'].str.contains('-')]

        if len(d) == 0:
            continue
        if time > 9:
            continue
        if time > 0:
            ax[r + 1,c].scatter((time - 1) * d['time interval (s)'].values[0] + 2,
                            d['area_normalized'].median(), edgecolor='dodgerblue',
                            facecolor='white',s=20)
        elif time == 0:
            ax[r + 1,c].scatter(0, d['area_normalized'].median(), 
                            edgecolor='dodgerblue',
                            facecolor='white',s=20)
    ax[r + 1,c].scatter(np.nan,np.nan,color='gray',marker='o', s=20,
                    label='individual unit cells')
    ax[r + 1,c].scatter(np.nan,np.nan,edgecolor='dodgerblue',facecolor='white',
                    s=20,label='median across unit cells')
    ax[r + 1,c].plot(t, area_contraction,color='tomato',ls='--',lw=2, 
                    label='pure contraction bound')
    ax[r + 1,c].legend(loc=1, fontsize=label_fontsize)
    ax[r + 1,c].set_ylabel('normalized area', fontsize=axis_fontsize)
    ax[r + 1,c].set_xlabel('time [s]', fontsize=axis_fontsize)
    ax[r + 1,c].set_ylim(0.5, 1.4)
    ax[r + 1,c].set_xlim(-2, 90)

    ax[r, c].set_title('%i sec' %(df['pb_time'].values[0]),
                       fontsize=axis_fontsize)
    n += 1

for a in ax.flatten():
    a.tick_params(axis='both', labelsize=axis_fontsize)

#ax[0,0].text(-14,1.0, '(A)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,1].text(-13,1.0,'(B)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,2].text(-13,1.0,'(C)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,3].text(-10,1.0,'(D)', ha='right', va='bottom', fontsize=label_fontsize)
fig.tight_layout()
plt.savefig('../../figures/SIFigX_contractionrate_areasize_ncd236replicates_1.pdf', bbox_inches='tight',
            facecolor='white')

n = 0
skip = 1
fig, ax = plt.subplots(n_rem,n_cols,figsize=(n_cols*2, n_rem*2))

for f,df in df_truncated.sort_values('pb_time').groupby('filename', sort=False):
    if (f != last_filename) and (skip == 1):
        continue
    elif (f == last_filename):
        skip = 0
    c = int(n % 5)
    r = 2 * int(n / 5)

    if r >= n_rows:
        last_filename = f
        break

    df_speedtrunc = df_contractionspeed[(df_contractionspeed['filename']==f)]

    df_compile = df_compilerates

    rad = np.linspace(0, df_truncated['radius'].max()*1.01,100)

    t = np.linspace(0,df_truncated[(df_truncated['time']<10)]['time'].max() * df_truncated['time interval (s)'].values[0],1000)

    #df = df_compiled[(df_compiled['filename']==data_root) & (~df_compiled['filename'].str.contains('-'))]
    
    contraction_median = df_compile['mean_rate'].values[0]
    area_contraction = (1 - contraction_median * t)**2

    ax[r,c].scatter(df_speedtrunc['radius'], df_speedtrunc['contraction speed'],
                color='gray', s=30, alpha=0.4, label='individual unit cells')
    contract_rate  = df_compile['mean_rate'].values[0]
    ax[r,c].plot(df_compile['distance'], df_compile['median'], 
                color='tomato', lw=3, ls='-', label=r'contraction rate = %.04f s$^{-1}$' %contract_rate)
    ax[r,c].fill_between(df_compile['distance'], df_compile['cred_low'], df_compile['cred_high'],
                        color='tomato',alpha=0.3,zorder=-2,label='95$\%$ credible region')
    ax[r,c].legend(loc=2, fontsize=label_fontsize)
    ax[r,c].set_ylabel('contraction speed [µm/s]', fontsize=axis_fontsize)
    ax[r,c].set_xlabel('radius [µm]', fontsize=axis_fontsize)
    ax[r,c].set_ylim(-0.01, 0.51)
    ax[r,c].set_xlim(np.min(rad)-0.01,np.max(rad))

    for id,d in df.groupby('cellID'):

        if d['time'].min() == 0:
            ax[r + 1,c].scatter((d[d['time']>0]['time'] - 1) * d[d['time']>0]['time interval (s)'] + 2,
                            d[d['time']>0]['area_normalized'], color='gray', s=10,
                            marker='o', alpha=0.1)
            ax[r + 1,c].scatter(0, d[d['time']==0]['area_normalized'], color='gray', s=10,
                            marker='o', alpha=0.1)
        else:
            d.loc[:,'time'] = d.loc[:,'time'] - d['time'].min()
            ax[r + 1,c].scatter((d['time']) * d['time interval (s)'],
                            d['area_normalized'], color='gray', s=10,
                            marker='o', alpha=0.1)
    # Compute  the mean area
    for time,d in df.groupby('time'):
        #d = d[~d['filename'].str.contains('-')]

        if len(d) == 0:
            continue
        if time > 9:
            continue
        if time > 0:
            ax[r + 1,c].scatter((time - 1) * d['time interval (s)'].values[0] + 2,
                            d['area_normalized'].median(), edgecolor='dodgerblue',
                            facecolor='white',s=20)
        elif time == 0:
            ax[r + 1,c].scatter(0, d['area_normalized'].median(), 
                            edgecolor='dodgerblue',
                            facecolor='white',s=20)
    ax[r + 1,c].scatter(np.nan,np.nan,color='gray',marker='o', s=20,
                    label='individual unit cells')
    ax[r + 1,c].scatter(np.nan,np.nan,edgecolor='dodgerblue',facecolor='white',
                    s=20,label='median across unit cells')
    ax[r + 1,c].plot(t, area_contraction,color='tomato',ls='--',lw=2, 
                    label='pure contraction bound')
    ax[r + 1,c].legend(loc=1, fontsize=label_fontsize)
    ax[r + 1,c].set_ylabel('normalized area', fontsize=axis_fontsize)
    ax[r + 1,c].set_xlabel('time [s]', fontsize=axis_fontsize)
    ax[r + 1,c].set_ylim(0.5, 1.4)
    ax[r + 1,c].set_xlim(-2, 90)

    ax[r, c].set_title('%i sec' %(df['pb_time'].values[0]),
                       fontsize=axis_fontsize)
    n += 1

for a in ax.flatten():
    a.tick_params(axis='both', labelsize=axis_fontsize)

#ax[0,0].text(-14,1.0, '(A)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,1].text(-13,1.0,'(B)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,2].text(-13,1.0,'(C)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,3].text(-10,1.0,'(D)', ha='right', va='bottom', fontsize=label_fontsize)
fig.tight_layout()
plt.savefig('../../figures/SIFigX_contractionrate_areasize_ncd236replicates_2.pdf', bbox_inches='tight',
            facecolor='white')

#%%