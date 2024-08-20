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

df_compiled = pd.read_csv('../../analyzed_data/unitcell_features_allmotors_v4.csv', sep=',')
df_compiled = df_compiled[(~df_compiled['filename'].str.contains('12-05')
                        & (~df_compiled['filename'].str.contains('11-30')))]
#df_compiled = df_allcompiled[~df_allcompiled['filename'].isin(df_avoid['filename'].values[:])]

# Truncate data to only unmerged unit cells
df_truncated = pd.DataFrame()
for fid,_d in df_compiled.groupby(['filename','cellID']):
    f,id = fid
    if ('12-05' in f) or ('11-30') in f:
        continue
    try:
        d_lastmerge = _d[:_d[_d['merged']==1.0].index[0]-1]
    except:
        d = _d
    else:
        d = d_lastmerge[d_lastmerge['merged']==0.0]
    if (len(d) <= 4) or (d['time'].min()>3):
        continue
    d['area_normalized'] = d['area']/d[d['time']==d['time'].min()]['area'].values[0]
    d['intensity_normalized'] = d['total_intensity']/d[d['time']==d['time'].min()]['total_intensity'].values[0]
    d['absolute_time'] = d['time'] * d['time interval (s)']
    df_truncated = pd.concat([df_truncated,d],ignore_index=True)


df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_v6.csv', sep=',')
df_compilerates = df_compilerates[df_compilerates['motor dilution']==1.0]

df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_v6.csv', sep=',')
df_contractionspeed = df_contractionspeed[df_contractionspeed['motor dilution']==1.0]

#%%
axis_fontsize = 30
label_fontsize = 48

motor_list = df_compilerates['motor'].unique()
x_pos = np.arange(0, len(motor_list), 1)

motor_speed = dict({'k401bac':250,'k401':600,
                    'ncd236':115, 'ncd281':90})
x_dict = dict({m:x for m,x in zip(motor_list,x_pos)})
#color_dict = dict({'ncd236' : 'rebeccapurple',
#                    'ncd281' : 'tomato'})

fig, ax = plt.subplots(1,1,figsize=(8,8))
for motor, _d in df_compilerates.groupby('motor'):
    d = _d[(_d['ATP (uM)']==1400) 
            & (_d['motor dilution']==1.0)
            & (_d['pluronic']==1)]
    ax.scatter(motor_speed[motor], d['mean_rate'].values[0], s=60, alpha=1.0, label=motor)
    ax.errorbar(motor_speed[motor], d['mean_rate'].values[0],
                yerr=np.array(d['mean_rate'].values[0] - d['rate_low'].values[0],
                                d['rate_high'].values[0] - d['mean_rate'].values[0]))
#ax.xaxis.set_tick_params(fontsize=24)
ax.set_xlabel('motor speed [nm/s]', fontsize=30)
ax.set_ylabel('contraction rate [1/s]', fontsize=30)
ax.legend(loc=4, fontsize=24)
#ax.set_xlim([0, 150])
#ax.set_ylim([0, 0.0025])
fig.tight_layout()
plt.savefig('../../figures/FigX_contraction_rates_motorspeeds.pdf', bbox_inches='tight',
            facecolor='white')
# %%
axis_fontsize = 30
label_fontsize = 24
fig, ax = plt.subplots(2,4,figsize=(32,16))
motor_list = ['ncd281', 'ncd236', 'k401bac', 'k401']
for n,motor in enumerate(motor_list):

    df_speedtrunc = df_contractionspeed[(df_contractionspeed['motor']==motor)
                & (df_contractionspeed['ATP (uM)']==1400)
                & (df_contractionspeed['motor dilution']==1.0)
                & (df_contractionspeed['pluronic']==1)]

    df_compile = df_compilerates[(df_compilerates['motor']==motor)
                                & (df_compilerates['motor dilution']==1.0)
                                & (df_compilerates['ATP (uM)']==1400)
                                & (df_compilerates['pluronic']==1)]

    r = np.linspace(0,df_truncated[df_truncated['motor']==motor]['radius'].max()*1.01,100)

    if motor=='ncd236':
        t = np.linspace(0,df_truncated[(df_truncated['motor']==motor) & (df_truncated['time']<10)]['time'].max() * df_truncated[df_truncated['motor']==motor]['time interval (s)'].values[0],1000)
    else:
        t = np.linspace(0,df_truncated[(df_truncated['motor']==motor)]['time'].max() * df_truncated[df_truncated['motor']==motor]['time interval (s)'].values[0],1000)

    #df = df_compiled[(df_compiled['filename']==data_root) & (~df_compiled['filename'].str.contains('-'))]
    
    contraction_median = df_compile['mean_rate'].values[0]
    area_contraction = (1 - contraction_median * t)**2

    ax[0,n].scatter(df_speedtrunc['radius'], df_speedtrunc['contraction speed'],
                color='gray', s=60, alpha=0.4, label='individual unit cells')
    contract_rate  = df_compile['mean_rate'].values[0]
    ax[0,n].plot(df_compile['distance'], df_compile['median'], 
                color='tomato', lw=3, ls='-', label=r'contraction rate = %.04f s$^{-1}$' %contract_rate)
    ax[0,n].fill_between(df_compile['distance'], df_compile['cred_low'], df_compile['cred_high'],
                        color='tomato',alpha=0.3,zorder=-2,label='95$\%$ credible region')
    ax[0,n].legend(loc=2, fontsize=24)
    ax[0,n].set_ylabel('contraction speed [µm/s]', fontsize=axis_fontsize)
    ax[0,n].set_xlabel('radius [µm]', fontsize=axis_fontsize)
    ax[0,n].set_ylim(-0.01, 1.01)
    ax[0,n].set_xlim(np.min(r)-0.01,np.max(r))

    for fid,d in df_truncated[df_truncated['motor']==motor].groupby(['filename','cellID']):
        f,id = fid


        if motor=='ncd236':
            ax[1,n].scatter((d[(d['time']<10) & (d['time']>0)]['time'] - 1) * d[(d['time']<10) & (d['time']>0)]['time interval (s)'] + 2,
                            d[(d['time']<10) & (d['time']>0)]['area_normalized'], color='gray', s=100,
                            marker='o', alpha=0.1)
            ax[1,n].scatter(0, d[(d['time']==0)]['area_normalized'], color='gray', s=100,
                            marker='o', alpha=0.1)
        else:
            ax[1,n].scatter((d[d['time']>0]['time'] - 1) * d[d['time']>0]['time interval (s)'] + 2,
                            d[d['time']>0]['area_normalized'], color='gray', s=40,
                            marker='o', alpha=0.1)
            ax[1,n].scatter(0, d[d['time']==0]['area_normalized'], color='gray', s=40,
                            marker='o', alpha=0.1)
    for time,d in df_truncated[df_truncated['motor']==motor].groupby('time'):
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
    ax[1,n].legend(loc=1, fontsize=24)
    ax[1,n].set_ylabel('normalized area', fontsize=axis_fontsize)
    ax[1,n].set_xlabel('time [s]', fontsize=axis_fontsize)

for a in ax.flatten():
    a.tick_params(axis='both', labelsize=axis_fontsize)

ax[1,2].set_xlim(-2, 30)
ax[1,3].set_xlim(-2, 40)

ax[0,0].set_title('(A) Ncd281', fontsize=axis_fontsize)
ax[0,1].set_title('(B) Ncd236 (original)', fontsize=axis_fontsize)
ax[0,2].set_title('(C) bacteria expressed K401', fontsize=axis_fontsize)
ax[0,3].set_title('(D) insect expressed K401', fontsize=axis_fontsize)
#ax[0,0].text(-14,1.0, '(A)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,1].text(-13,1.0,'(B)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,2].text(-13,1.0,'(C)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,3].text(-10,1.0,'(D)', ha='right', va='bottom', fontsize=label_fontsize)
fig.tight_layout()
plt.savefig('../../figures/FigX_contractionrate_areasize_allmotors_corrected.pdf', bbox_inches='tight',
            facecolor='white')

# %%
df_comsol = pd.read_csv('../../analyzed_data/comsol_advdiff_areameans.csv',
                        sep=',')

df_comsol = df_comsol[df_comsol['merged']==0]

axis_fontsize = 30
label_fontsize = 24
fig, ax = plt.subplots(1,4,figsize=(32,8))
motor_list = ['ncd281', 'ncd236', 'k401bac', 'k401']
df_fullatp = df_truncated[(df_truncated['ATP (uM)'] == 1400) & (df_truncated['pluronic'] == 1)]
for n,motor in enumerate(motor_list):

    r = np.linspace(0, df_fullatp[df_fullatp['motor']==motor]['radius'].max()*1.01,100)
    
    if motor=='ncd236':
        t = np.linspace(0, df_fullatp[(df_fullatp['motor']==motor) & (df_fullatp['time']<10)]['time'].max() * df_fullatp[df_fullatp['motor']==motor]['time interval (s)'].values[0],1000)
    else:
        t = np.linspace(0, df_fullatp[(df_fullatp['motor']==motor)]['time'].max() * df_fullatp[df_fullatp['motor']==motor]['time interval (s)'].values[0],1000)

    #df = df_compiled[(df_compiled['filename']==data_root) & (~df_compiled['filename'].str.contains('-'))]
    
    contraction_median = df_compilerates[(df_compilerates['motor']==motor) 
                                        & (df_compilerates['ATP (uM)']==1400)
                                        & (df_compilerates['pluronic']==1)]['mean_rate'].values[0]
    area_contraction = (1 - contraction_median * t)**2

    for time,d in df_fullatp[df_fullatp['motor']==motor].groupby('time'):
        #d = d[~d['filename'].str.contains('-')]

        if len(d) == 0:
            continue
        if (motor=='ncd236') or (motor=='k401'):
            if time > 9:
                continue
        elif (motor=='k401bac'):
            if time > 7:
                continue
        if time > 0:
            ax[n].errorbar((time - 1) * d['time interval (s)'].values[0] + 2,
                            d['area_normalized'].mean(), 
                            yerr = d['area_normalized'].std(), 
                            marker = 'o', color='dodgerblue', ms=10, lw=2)
        elif time == 0:
            ax[n].errorbar(0, d['area_normalized'].mean(), 
                            yerr = d['area_normalized'].std(),
                            marker = 'o', color='dodgerblue', ms=10, lw=2)

    ax[n].scatter(np.nan,np.nan,edgecolor='dodgerblue',facecolor='white',
                    s=100,label='mean area')

    ax[n].plot(t, area_contraction,color='tomato',ls='--',lw=2, 
                label='pure contraction bound')

    alpha = float('%.4f' %contraction_median)
    d_comsol = df_comsol[df_comsol['alpha'] == alpha]

    if len(d_comsol) == 0:
        print('no COMSOL data')
    else:
        dt = df_fullatp[df_fullatp['motor']==motor]['time interval (s)'].values[0]
        time_array = d_comsol[(d_comsol['time'] <= time * dt)]['time'].unique()
        time_array = np.sort(time_array)

        low_mean = np.zeros(len(time_array))
        high_mean = np.zeros(len(time_array))

        for time_c in time_array:
            if motor=='ncd236':
                lowindex = 2
            elif motor=='k401bac':
                lowindex = 0
            else:
                lowindex = 0
            low_mean[np.argwhere(time_array==time_c)[0]] = d_comsol[(d_comsol['time'] == time_c)
                                                                & (d_comsol['D'] == d_comsol['D'].unique()[lowindex])]['normalized_area'].mean()
            high_mean[np.argwhere(time_array==time_c)[0]] = d_comsol[(d_comsol['time'] == time_c)
                                                                & (d_comsol['D'] == d_comsol['D'].unique()[lowindex+1])]['normalized_area'].mean()

        ax[n].fill_between(time_array, low_mean, high_mean,
                            color='tomato', alpha=0.5,
                            label=r'%.01f $\times 10^{-3}$ µm$^2$/s < D < %.01f $\times 10^{-3}$ µm$^2$/s' %(d_comsol['D'].unique()[lowindex] * 1000, d_comsol['D'].unique()[lowindex+1] * 1000))
    if motor == 'ncd236':
        ax[n].set_xlim(-5,95)
    elif motor == 'k401bac':
        ax[n].set_xlim(-2, 35)
    elif motor == 'k401':
        ax[n].set_xlim([-2, 50])

    ax[n].legend(loc=3, fontsize=20)
    ax[n].set_ylabel('normalized area', fontsize=axis_fontsize)
    ax[n].set_xlabel('time [s]', fontsize=axis_fontsize)

for a in ax.flatten():
    a.tick_params(axis='both', labelsize=axis_fontsize)

ax[0].set_title('(A) Ncd281', fontsize=axis_fontsize)
ax[1].set_title('(B) Ncd236 (original)', fontsize=axis_fontsize)
ax[2].set_title('(C) bacteria expressed K401', fontsize=axis_fontsize)
ax[3].set_title('(D) insect expressed K401', fontsize=axis_fontsize)
#ax[0,0].text(-14,1.0, '(A)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,1].text(-13,1.0,'(B)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,2].text(-13,1.0,'(C)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,3].text(-10,1.0,'(D)', ha='right', va='bottom', fontsize=label_fontsize)
fig.tight_layout()
#plt.savefig('../../figures/FigX_areasize_exptcomsolcompare_allmotors.pdf', bbox_inches='tight',
#            facecolor='white')
#%%
# %%
axis_fontsize = 30
label_fontsize = 24
fig, ax = plt.subplots(1,1,figsize=(8,8))
motor_list = ['ncd281', 'ncd236', 'k401bac', 'k401']
colorlist = ['dodgerblue', 'tomato', 'rebeccapurple', 'green']
df_fullatp = df_truncated[(df_truncated['ATP (uM)'] == 1400) & (df_truncated['pluronic'] == 1)]

df_fullatp['absolute_time'] = (df_fullatp['time'] - 1) * df_fullatp['time interval (s)'] + 2
df_fullatp.loc[df_fullatp['absolute_time'] < 0, 'absolute_time'] = 0

for n,motor in enumerate(motor_list):

    #df = df_compiled[(df_compiled['filename']==data_root) & (~df_compiled['filename'].str.contains('-'))]
    
    contraction_median = df_compilerates[(df_compilerates['motor']==motor) 
                                        & (df_compilerates['ATP (uM)']==1400)
                                        & (df_compilerates['pluronic']==1)]['mean_rate'].values[0]
    area_contraction = (1 - contraction_median * t)**2
    alpha = float('%.4f' %contraction_median)
    d_comsol = df_comsol[df_comsol['alpha'] == alpha]

    df_stats = pd.DataFrame()

    for time,d in df_fullatp[df_fullatp['motor']==motor].groupby('time'):

        if len(d) == 0:
            continue
        if (motor=='ncd236') or (motor=='k401'):
            if time > 9:
                continue
        elif (motor=='k401bac'):
            if time > 7:
                continue

        _df_stats = pd.DataFrame([[d['absolute_time'].values[0], d['area_normalized'].mean(), d['area_normalized'].std()]],
                                columns=['time', 'mean_area', 'std_area'])

        df_stats = pd.concat([df_stats, _df_stats], ignore_index=True)


    ax.errorbar(df_stats['time'] * alpha,
                df_stats['mean_area'], 
                yerr = df_stats['std_area'], color=colorlist[n],
                marker = 'o', ms=5, linestyle='-', lw=1)

    #ax[n].scatter(np.nan,np.nan,edgecolor='dodgerblue',facecolor='white',
    #                s=100,label='mean area')

    #ax[n].plot(alpha * t, area_contraction, color='tomato',ls='--',lw=2, 
    #            label='pure contraction bound')

    """if len(d_comsol) == 0:
        print('no COMSOL data')
    else:
        dt = df_fullatp[df_fullatp['motor']==motor]['time interval (s)'].values[0]
        time_array = d_comsol[(d_comsol['time'] <= time * dt)]['time'].unique()
        time_array = np.sort(time_array)

        low_mean = np.zeros(len(time_array))
        high_mean = np.zeros(len(time_array))

        for time_c in time_array:
            if motor=='ncd236':
                lowindex = 2
            elif motor=='k401bac':
                lowindex = 0
            else:
                lowindex = 0
            low_mean[np.argwhere(time_array==time_c)[0]] = d_comsol[(d_comsol['time'] == time_c)
                                                                & (d_comsol['D'] == d_comsol['D'].unique()[lowindex])]['normalized_area'].mean()
            high_mean[np.argwhere(time_array==time_c)[0]] = d_comsol[(d_comsol['time'] == time_c)
                                                                & (d_comsol['D'] == d_comsol['D'].unique()[lowindex+1])]['normalized_area'].mean()

        ax[n].fill_between(time_array, low_mean, high_mean,
                            color='tomato', alpha=0.5,
                            label=r'%.01f $\times 10^{-3}$ µm$^2$/s < D < %.01f $\times 10^{-3}$ µm$^2$/s' %(d_comsol['D'].unique()[lowindex] * 1000, d_comsol['D'].unique()[lowindex+1] * 1000))"""
    #ax[n].set_xlim([-2, 50])

ax.legend(loc=3, fontsize=16)
ax.set_ylabel('normalized area', fontsize=16)
ax.set_xlabel(r'dimensionless time ($\tau = \alpha t$) [s]', fontsize=16)

ax.tick_params(axis='both', labelsize=16)

#ax[0,0].text(-14,1.0, '(A)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,1].text(-13,1.0,'(B)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,2].text(-13,1.0,'(C)', ha='right', va='bottom', fontsize=label_fontsize)
#ax[0,3].text(-10,1.0,'(D)', ha='right', va='bottom', fontsize=label_fontsize)
fig.tight_layout()
plt.savefig('../../figures/FigX_areasize_allmotors_dimensionless.pdf', bbox_inches='tight',
            facecolor='white')

#%%
# Plotting area against intensity
axis_fontsize = 30
label_fontsize = 24
fig, ax = plt.subplots(1,1,figsize=(8,8))
motor_list = ['ncd281', 'ncd236', 'k401bac', 'k401']
#colorlist = ['dodgerblue', 'tomato', 'rebeccapurple', 'green']
df_fullatp = df_truncated[(df_truncated['ATP (uM)'] == 1400) & (df_truncated['pluronic'] == 1)]

df_fullatp['absolute_time'] = (df_fullatp['time'] - 1) * df_fullatp['time interval (s)'] + 2
df_fullatp.loc[df_fullatp['absolute_time'] < 0, 'absolute_time'] = 0

for n,motor in enumerate(motor_list):

    if motor!='ncd236':
        continue

    for time,d in df_fullatp[df_fullatp['motor']==motor].groupby('time'):
        realtime = d['absolute_time'].values[0]
        ax.scatter(d['intensity_normalized'] / d['area_normalized'], d['area_normalized'], 
                    marker = 'o', s=20, alpha=0.5, label='t=%i' %realtime)

ax.legend(loc=3, fontsize=16)
ax.set_ylabel('normalized area', fontsize=16)
ax.set_xlabel('density', fontsize=16)
ax.legend(loc=1)

#ax.set_xlim(0.9,1.1)
#ax.set_ylim(0.5,1.1)

ax.tick_params(axis='both', labelsize=16)
fig.tight_layout()
#plt.savefig('../../figures/FigX_normalizeddensity_v_normalizedarea.pdf',
#            bbox_inches='tight', facecolor='white')
# %%
# Plotting area against intensity
axis_fontsize = 30
label_fontsize = 24
fig, ax = plt.subplots(2,4,figsize=(32,16))
motor_list = ['ncd281', 'ncd236', 'k401bac', 'k401']
#colorlist = ['dodgerblue', 'tomato', 'rebeccapurple', 'green']
df_fullatp = df_truncated[(df_truncated['ATP (uM)'] == 1400) & (df_truncated['pluronic'] == 1)]

df_fullatp['absolute_time'] = (df_fullatp['time'] - 1) * df_fullatp['time interval (s)'] + 2
df_fullatp.loc[df_fullatp['absolute_time'] < 0, 'absolute_time'] = 0

for time,d in df_fullatp[df_fullatp['motor']=='ncd236'].groupby('time'):
    if time == 0:
        continue

    n = int((time - 1) / 4)
    m = int((time - 1) % 4)

    if n > 1:
        break

    realtime = d['absolute_time'].values[0]
    ax[n,m].scatter(d['total_intensity'] / d['area_normalized'], d['area_normalized'], 
                    marker = 'o', s=20, alpha=0.5)
    ax[n,m].set_title('t = %i sec' %realtime, fontsize=label_fontsize)

for a in ax.ravel():
    a.set_ylabel('normalized area', fontsize=16)
    a.set_xlabel('density', fontsize=16)
    a.tick_params(axis='both', labelsize=16)

#ax.set_xlim(0.9,1.1)
#ax.set_ylim(0.5,1.1)

fig.tight_layout()
plt.savefig('../../figures/FigX_normalizeddensity_v_normalizedarea_individual.pdf',
            bbox_inches='tight', facecolor='white')
# %%
