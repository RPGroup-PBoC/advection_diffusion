#%%
# Compiling figure from data analysis of unit cells
import os
import sys
sys.path.insert(0,'../')
import active_matter_pkg as amp
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
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

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_multithreading.csv', sep=',')
df_compilerates = df_compilerates[df_compilerates['motor dilution']==1.0]


df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
df_contractionspeed = df_contractionspeed[df_contractionspeed['motor dilution']==1.0]

# %%
motor = 'k401bac'
axis_fontsize = 30
label_fontsize = 24

df_atp = df_truncated[(df_truncated['motor']==motor) 
                      & (df_truncated['pluronic']==1) 
                      & (df_truncated['ATP (uM)'].isin([25, 100, 500]))].sort_values('ATP (uM)')

num_atp = np.arange(0, len(df_atp['ATP (uM)'].unique()), 1)
atp_dict = dict({atp:n for atp,n in zip(df_atp['ATP (uM)'].unique(), num_atp)})
letter_dict = dict({atp:letter for atp,letter in zip(df_atp['ATP (uM)'].unique(), ['A','B','C'])})

fig, ax = plt.subplots(2, len(num_atp),figsize=(8 * len(num_atp),16))


for atp, _d in df_atp.groupby('ATP (uM)'):
    n = atp_dict[atp]
    r = np.linspace(0, _d['radius'].max()*1.01,100)

    if motor=='ncd236':
        t = np.linspace(0, _d[(_d['time']<10)]['time'].max() * _d['time interval (s)'].values[0],1000)

    df_comprates = df_compilerates[(df_compilerates['motor']==motor)
                                    & (df_compilerates['ATP (uM)']==atp)
                                    & (df_compilerates['pluronic']==1)]
    
    contraction_median = df_compilerates[(df_compilerates['motor']==motor) 
                                         & (df_compilerates['ATP (uM)']==atp) 
                                         & (df_compilerates['pluronic']==1)]['mean_rate'].values[0]
    area_contraction = (1 - contraction_median * t)**2

    df_contract = df_contractionspeed[(df_contractionspeed['motor']==motor) 
                                      & (df_contractionspeed['ATP (uM)']==atp) 
                                      & (df_contractionspeed['pluronic']==1)]

    ax[0,n].scatter(df_contract[(df_contract['filename']!=data_root)]['radius'],
                    df_contract[(df_contract['filename']!=data_root)]['contraction speed'],
                    color='gray', s=60, alpha=0.4, label='individual unit cells')
    contract_rate  = df_comprates['mean_rate'].values[0]
    ax[0,n].plot(df_comprates['distance'], df_comprates['median'], 
                color='tomato', lw=3, ls='-', label=r'contraction rate = %.04f s$^{-1}$' %contract_rate)
    ax[0,n].fill_between(df_comprates['distance'], df_comprates['cred_low'], df_comprates['cred_high'],
                        color='tomato',alpha=0.3,zorder=-2,label='95$\%$ credible region')
    ax[0,n].legend(loc=2, fontsize=24)
    ax[0,n].set_ylabel('contraction speed [µm/s]', fontsize=axis_fontsize)
    ax[0,n].set_xlabel('radius [µm]', fontsize=axis_fontsize)
    ax[0,n].set_ylim(-0.01, 1.01)
    ax[0,n].set_xlim(np.min(r)-0.01,np.max(r))

    for fid,d in _d.groupby(['filename','cellID']):
        f,id = fid

        if motor=='ncd236':
            ax[1,n].scatter((d[(d['time']<10) & (d['time']>0)]['time'] - 1) * d[(d['time']<10) & (d['time']>0)]['time interval (s)'] + 2,
                            d[(d['time']<10) & (d['time']>0)]['area_normalized'], color='gray', s=100,
                            marker='o', alpha=0.1)
            ax[1,n].scatter((d[(d['time']==0)]['time']) * d[(d['time']==0)]['time interval (s)'] + 2,
                            d[(d['time']==0)]['area_normalized'], color='gray', s=100,
                            marker='o', alpha=0.1)
        else:
            ax[1,n].scatter((d[d['time']>0]['time'] - 1) * d[d['time']>0]['time interval (s)'] + 2,
                            d[d['time']>0]['area_normalized'], color='gray', s=40,
                            marker='o', alpha=0.1)
            ax[1,n].scatter(d[d['time']==0]['time'] * d[d['time']==0]['time interval (s)'],
                            d[d['time']==0]['area_normalized'], color='gray', s=40,
                            marker='o', alpha=0.1)
    for time,d in _d.groupby('time'):
        #d = d[~d['filename'].str.contains('-')]

        if len(d) == 0:
            continue
        if motor=='ncd236':
            if time > 9:
                continue
        if time > 0:
            ax[1,n].scatter((time - 1)*d['time interval (s)'].values[0] + 2,
                            d['area_normalized'].mean(), edgecolor='dodgerblue',
                            facecolor='white',s=100)
        elif time == 0:
            ax[1,n].scatter(0, d['area_normalized'].mean(), 
                            edgecolor='dodgerblue',
                            facecolor='white',s=100)
    ax[1,n].scatter(np.nan,np.nan,color='gray',marker='o', s=100, 
                    label='individual unit cells')
    ax[1,n].scatter(np.nan,np.nan,edgecolor='dodgerblue',facecolor='white',
                    s=100,label='mean across unit cells')
    ax[1,n].plot(t, area_contraction,color='tomato',ls='--',lw=2, 
                label='pure contraction bound')
    ax[1,n].set_ylim(0.6, 1.2)
    if n==2:
        ax[1,n].set_xlim(-2, 30)
    else:
        ax[1,n].set_xlim(-2, 60)
    ax[1,n].legend(loc=1, fontsize=24)
    ax[1,n].set_ylabel('normalized area', fontsize=axis_fontsize)
    ax[1,n].set_xlabel('time [s]', fontsize=axis_fontsize)
    ax[0,n].set_title('(%s) %i uM ATP' % (letter_dict[atp], atp), fontsize=axis_fontsize)

for a in ax.flatten():
    a.tick_params(axis='both', labelsize=axis_fontsize)

fig.tight_layout()
#plt.savefig('../../figures/FigX_ATP_%s.pdf' %motor, bbox_inches='tight', facecolor='white')
# %%
# Cycle through conditions
motor = 'ncd236'
pluronic = 1
atp = 75

corner_pad = 20

#num_array = np.array([2,4,5,3,11,8,10,12,9,13,14,17,15,16,21,19,23,20,22,25,26])
#map_list = np.arange(1, len(num_array)+1, 1)

#number_mapping = dict({a:b for a,b in zip(num_array,map_list)})

df_speedtrunc = df_contractionspeed[(df_contractionspeed['motor']==motor)
                & (df_contractionspeed['ATP (uM)']==atp)
                & (df_contractionspeed['motor dilution']==1.0)
                & (df_contractionspeed['pluronic']==pluronic)]

df_compile = df_compilerates[(df_compilerates['motor']==motor)
                            & (df_compilerates['motor dilution']==1.0)
                            & (df_compilerates['ATP (uM)']==atp)
                            & (df_compilerates['pluronic']==pluronic)]

files = df_truncated[(df_truncated['motor']==motor)
                    & (df_truncated['ATP (uM)']==atp)
                    & (df_truncated['pluronic']==pluronic)]['filename'].unique()

r = np.linspace(0, df_compile['distance'].max() * 1.01, 100)
t = np.linspace(0, 90, 1000)
dt = df_info['time interval (s)'].values[0]

for f in files:

    df = df_compiled[df_compiled['filename']==f]

    contraction_median = df_compile['mean_rate'].values[0]
    area_contraction = (1 - contraction_median * t)**2

    axis_fontsize = 30
    label_fontsize = 48
    fig, ax = plt.subplots(1,3,figsize=(24,8))

    _,_,mt_trimmed,_,_ = amp.io.tiff_walk(f)
    im = io.imread(mt_trimmed[0])
    #Place scale bar in im2
    im[-corner_pad-10:-corner_pad,int(-corner_pad-50/um_per_pxl):-corner_pad] = im.max()

    axis_fontsize = 30
    label_fontsize = 48
    ax[0].imshow(im)
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[0].text(int(np.shape(im)[0]-corner_pad-25/um_per_pxl), np.shape(im)[0]-corner_pad-20, '50 μm',
                fontsize=30, ha='center', va='bottom', color='white', weight='bold')
    ax[0].set_title(os.path.split(f)[-1])

    for id,d in df.groupby('cellID'):
        if (len(d) <= 4) or (d['time'].min()>3) or (d['radius'].values[0]>125):
            continue

        d['absolute_time'] = (d['time'] - 1) * dt + 2
        d.loc[d['absolute_time'] < 0, 'absolute_time'] = 0
        ax[1].plot(d['absolute_time'], d['radius'],
                    color='rebeccapurple', ls='-', lw=2, marker='o')
    
    ax[1].set_ylabel('distance from center of\ncontraction [μm]', fontsize=axis_fontsize)
    ax[1].set_xlabel('time [s]', fontsize=axis_fontsize)

    for file,d in df_speedtrunc.groupby('filename'):
        if file==f:
            ax[2].scatter(d['radius'],d['contraction speed'], facecolor='white', lw=2,
                        marker='P', edgecolor='rebeccapurple', s=80, alpha=1.0)
        else:
            ax[2].scatter(d['radius'],d['contraction speed'],
                        color='gray', s=60, alpha=0.4)
    ax[2].scatter(np.nan,np.nan,color='gray',alpha=0.4,label='individual unit cells')
    ax[2].scatter(np.nan,np.nan,facecolor='white',edgecolor='rebeccapurple', 
                marker='P', s=70, alpha=1.0, label='sample data', lw=2)
    ax[2].plot(df_compile['distance'],
                df_compile['median'], 
                color='tomato', lw=3, ls='-', 
                label=r'contraction rate = %.04f s$^{-1}$' %df_compile['mean_rate'].values[0])
    ax[2].fill_between(df_compile['distance'],
                    df_compile['cred_low'],
                    df_compile['cred_high'],
                        color='tomato',alpha=0.3,zorder=-2,label='95$\%$ credible region')
    ax[2].legend(loc=2, fontsize=18)
    ax[2].set_ylabel('contraction speed [µm/s]', fontsize=axis_fontsize)
    ax[2].set_xlabel('radius [µm]', fontsize=axis_fontsize)
    ax[2].set_ylim([-0.02,1.0])
    ax[2].set_xlim(-0.1,120)

    fig.tight_layout()
    plt.show()
# %%
df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_separatereplicates.csv', sep=',')

fig, ax = plt.subplots(1,1,figsize=(8,8))
for fa,_d in df_compilerates[(df_compilerates['motor']=='ncd236') & (df_compilerates['pluronic']==1)].groupby(['filename','ATP (uM)']):
    f, atp = fa
    
    if atp == 300:
        continue
    elif (atp > 30) and (atp < 300):
        if ('02-15' not in f) and ('02-14' not in f):
            continue
    
    ax.scatter(atp, _d['mean_rate'].values[0],  marker='o', s=20)
    #ax.errorbar(atp, _d['mean_rate'].values[0], 
    #            yerr=np.array(_d['mean_rate'].values[0] - _d['rate_low'].values[0],
    #                        _d['rate_high'].values[0] - _d['mean_rate'].values[0]),
    #            color='black', marker='o', ms=10)
ax.set_xlim([0, 1500])
ax.set_ylim([0, 0.0040])
ax.set_xlabel('ATP concentration [µM]', fontsize=20)
ax.set_ylabel('contraction rate [1/s]', fontsize=20)
ax.set_title('Ncd236', fontsize=20)
# %%
fig, ax = plt.subplots(1,1,figsize=(8,8))
for atp,_d in df_compilerates[(df_compilerates['motor']=='ncd236') & (df_compilerates['pluronic']==1)].groupby(['ATP (uM)']):
    
    if atp == 300:
        continue
    elif (atp > 30) and (atp < 300):
        _d = _d[(_d['filename'].str.contains('02-15')) | (~_d['filename'].str.contains('02-14')) | (~_d['filename'].str.contains('02-16'))]
    ax.scatter(atp, np.mean(_d['mean_rate'].unique()),  marker='o', s=20)
    #ax.errorbar(atp, _d['mean_rate'].values[0], 
    #            yerr=np.array(_d['mean_rate'].values[0] - _d['rate_low'].values[0],
    #                        _d['rate_high'].values[0] - _d['mean_rate'].values[0]),
    #            color='black', marker='o', ms=10)
ax.set_xlim([3, 3000])
ax.set_ylim([0, 0.0040])
ax.set_xscale('log')
ax.set_xlabel('ATP concentration [µM]', fontsize=20)
ax.set_ylabel('contraction rate [1/s]', fontsize=20)
ax.set_title('Ncd236', fontsize=20)
# %%
axis_fontsize = 30
label_fontsize = 24
fig, ax = plt.subplots(1,1,figsize=(8,8))
import matplotlib

color = matplotlib.colormaps['viridis']
x = np.linspace(0.4, 1.0, len(df_compilerates[df_compilerates['motor']=='ncd236']['ATP (uM)'].unique()))
colors = [color(_x) for _x in x]
atp_list = df_compilerates[df_compilerates['motor']=='ncd236']['ATP (uM)'].unique()
color_dict = dict(zip(atp_list,colors))

for n,atp in enumerate(atp_list):

    df_compile = df_compilerates[(df_compilerates['motor']=='ncd236')
                                & (df_compilerates['motor dilution']==1.0)
                                & (df_compilerates['ATP (uM)']==atp)
                                & (df_compilerates['pluronic']==1)]
    contraction_median = df_compile['mean_rate'].values[0]

    r = np.linspace(0,df_truncated[(df_truncated['motor']=='ncd236') & (df_truncated['ATP (uM)']==atp)]['radius'].max()*1.01,100)

    for time,d in df_truncated[(df_truncated['motor']=='ncd236') & (df_truncated['ATP (uM)']==atp)].groupby('time'):
        #d = d[~d['filename'].str.contains('-')]

        if len(d) == 0:
            continue
        if time > 9:
            continue
        if time > 0:
            ax.scatter(contraction_median * ((time - 1) * d['time interval (s)'].values[0] + 2),
                        d['area_normalized'].mean(), color=color_dict[atp],
                        s=100)
        elif time == 0:
            ax.scatter(0, d['area_normalized'].mean(), 
                        color=color_dict[atp], s=100)

    ax.scatter(np.nan,np.nan,color=color_dict[atp],
                    s=100,label=atp)

ax.legend(loc=1, fontsize=24)
ax.set_ylabel('normalized area', fontsize=axis_fontsize)
ax.set_xlabel('nondimensionalized time', fontsize=axis_fontsize)

ax.tick_params(axis='both', labelsize=axis_fontsize)
ax.set_ylim(0.5, 1.4)
fig.tight_layout()
#plt.savefig('../../figures/FigX_contractionrate_areasize_allmotors_nondimensionalized.pdf', bbox_inches='tight',
#            facecolor='white')
# %%
