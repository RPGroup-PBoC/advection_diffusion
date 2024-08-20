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

df_compiled = pd.read_csv('../../analyzed_data/unitcell_features_compiled.csv', sep=',')
#df_compiled = pd.read_csv('../../analyzed_data/pluronic_dataset.csv', sep=',')
df_truncated = df_compiled[(~df_compiled['filename'].str.contains('12-05')
                        & (~df_compiled['filename'].str.contains('11-30')))]

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_pluronic.csv', sep=',')
df_compilerates = df_compilerates[df_compilerates['motor dilution']==1.0]

df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
df_contractionspeed = df_contractionspeed[df_contractionspeed['motor dilution']==1.0]

df_comsol = pd.read_csv('../../analyzed_data/comsol_unitcell_compiled.csv', sep=',')

df_fits = pd.read_csv('../../analyzed_data/diffusion_fits.csv', sep=',')
df_fits = df_fits[df_fits['ATP (uM)']==1400]
# %%
t = np.linspace(0,200,1000)
motor = 'k401bac'
df = df_truncated[(df_truncated['motor']==motor) & (df_truncated['ATP (uM)']==1400)]
axis_fontsize = 30
label_fontsize = 24
num_sets = len(df['pluronic'].unique())
fig, ax = plt.subplots(2,num_sets,figsize=(8 * num_sets,16))
for pluronic, _d in df.groupby('pluronic'):
    if pluronic == 0:
        n = 0
    elif pluronic == 0.1:
        n = 1
    elif pluronic == 1:
        n = 2
    elif pluronic == 3:
        n = 3
    elif pluronic == 10:
        n = 4

    r = np.linspace(0, _d['radius'].max() * 1.01, 100)

    df_comprates = df_compilerates[(df_compilerates['motor']==motor) 
                                   & (df_compilerates['pluronic']==pluronic) 
                                   & (df_compilerates['ATP (uM)']==1400)]
    
    contraction_median = df_compilerates[(df_compilerates['motor']==motor)
                                        & (df_compilerates['pluronic']==pluronic)
                                        & (df_compilerates['ATP (uM)']==1400)]['mean_rate'].values[0]
    area_contraction = (1 - contraction_median * t)**2

    df_contract = df_contractionspeed[(df_contractionspeed['motor']==motor) 
                                      & (df_contractionspeed['pluronic']==pluronic) 
                                      & (df_contractionspeed['ATP (uM)']==1400)]

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
    ax[1,n].legend(loc=3, fontsize=24)
    ax[1,n].set_ylabel('normalized area', fontsize=axis_fontsize)
    ax[1,n].set_xlabel('time [s]', fontsize=axis_fontsize)
    ax[0,n].set_title('%ix pluronic' %pluronic, fontsize=axis_fontsize)

for a in ax[0,:]:
    a.set_xlim(0, 110)
    a.set_ylim(0, 1.0)

for a in ax[1,:]:
    a.set_xlim(0, 35)
    a.set_ylim(0.4, 1.1)

for a in ax.flatten():
    a.tick_params(axis='both', labelsize=axis_fontsize)

fig.tight_layout()
#plt.savefig('../../figures/FigX_pluronic_%s.pdf' %motor, bbox_inches='tight', facecolor='white')
#%%
motor = 'k401bac'
axis_fontsize = 15
label_fontsize = 24

df = df_compilerates[(df_compilerates['motor']==motor)
                     & (df_compilerates['ATP (uM)']==1400)]
motor_list = df['motor'].unique()
x_pos = np.arange(0, len(motor_list), 1)

x_dict = dict({m:x for m,x in zip(motor_list,x_pos)})
#color_dict = dict({'ncd236' : 'rebeccapurple',
#                    'ncd281' : 'tomato'})

fig, ax = plt.subplots(1,2,figsize=(9,4))
for pluronic, _d in df.groupby('pluronic'):
    d = _d[(_d['ATP (uM)']==1400) 
            & (_d['motor dilution']==1.0)]
    if pluronic == 0:
        pluronic += 2 * 10**(-3)
    ax[0].scatter(pluronic * 0.5, d['mean_rate'].values[0], s=90, alpha=1.0, 
               color='dodgerblue', label='%.2d mg/mL' %(pluronic * 0.5))
    ax[0].errorbar(pluronic * 0.5, d['mean_rate'].values[0],
                yerr=np.array(d['mean_rate'].values[0] - d['rate_low'].values[0],
                                d['rate_high'].values[0] - d['mean_rate'].values[0]),
                color='dodgerblue')
    
_df_fits = df_fits[df_fits['motor']=='k401bac']
_df_fits[_df_fits['pluronic']==_df_fits['pluronic'].min()] = 2 * 10**(-3)
ax[1].scatter(_df_fits['pluronic'] * 0.5, _df_fits['D (1st quartile)'], 
                        color='k', marker='^', s=90, linestyle='None',
                        label='1st quartile')
ax[1].scatter(_df_fits['pluronic'] * 0.5, _df_fits['D (median)'], 
                color='rebeccapurple', marker='o', s=80, linestyle='None',
                label='median')
ax[1].scatter(_df_fits['pluronic'] * 0.5, _df_fits['D (3rd quartile)'], 
                color='green', marker='P', s=90, linestyle='None',
                label='3rd quartile')

#ax.xaxis.set_tick_params(fontsize=24)
ax[0].set_xlabel('pluronic concentration [mg/mL]', fontsize=16)
ax[0].set_ylabel('contraction rate [1/s]', fontsize=16)
ax[1].set_xlabel('pluronic concentration [mg/mL]', fontsize=16)
ax[1].set_ylabel(r'diffusion constant [µm$^2$/s]', fontsize=16)
ax[1].set_ylim([0, 0.01])
ax[1].legend(loc=2)

ax[0].set_xscale('log')
ax[1].set_xscale('log')

for a in ax:
    a.set_xlim([7 * 10**(-4), 10])
    a.set_xticks([10**(-3), 0.05, 0.5, 1.5, 5])
    a.set_xticklabels([0, 0.05, 0.5, 1.5, 5])

ax[0].text(0.00003, 0.0105, '(A)', fontsize=20)
ax[1].text(0.00003, 0.0103, '(B)', fontsize=20)

fig.tight_layout()
#plt.savefig('../../figures/FigX_contraction_rates_pluronic_%s.pdf' %motor, 
#            bbox_inches='tight', facecolor='white')
# %%
axis_fontsize = 30
label_fontsize = 48

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_separatereplicates.csv', sep=',')

fig, ax = plt.subplots(1,1,figsize=(8,8))
for fp,_d in df_compilerates[(df_compilerates['motor']=='k401bac') & (df_compilerates['ATP (uM)']==1400)].groupby(['filename','pluronic']):
    f, pluronic = fp
    
    ax.scatter(pluronic * 0.5, _d['mean_rate'].values[0],  marker='o', s=20, color='gray')

for pluronic,_d in df_compilerates[(df_compilerates['motor']=='k401bac') & (df_compilerates['ATP (uM)']==1400)].groupby('pluronic'):
    
    ax.scatter(pluronic * 0.5, _d['mean_rate'].median(),  marker='o', s=20, edgecolor='dodgerblue', facecolor='white')
    ax.errorbar(pluronic * 0.5, _d['mean_rate'].median(),
                yerr=[[_d['rate_high'].median() - _d['mean_rate'].median()],
                        [_d['mean_rate'].median() - _d['rate_low'].median()]],
                marker='o', ms=5, lw=2, color='dodgerblue')
    
ax.set_xlim([-0.5, 6])
ax.set_ylim([0, 0.020])
ax.set_xscale('linear')
ax.set_xlabel('pluronic concentration [mg/mL]', fontsize=20)
ax.set_ylabel('contraction rate [1/s]', fontsize=20)
ax.set_title('K401 (bacterial)', fontsize=20)
# %%

