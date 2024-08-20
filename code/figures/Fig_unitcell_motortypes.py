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
df_truncated = df_compiled[(~df_compiled['filename'].str.contains('12-05')
                        & (~df_compiled['filename'].str.contains('11-30')))]
df_truncated = df_truncated[(df_truncated['ATP (uM)']==1400) & (df_truncated['pluronic']==1)]

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_multithreading.csv', sep=',')
df_compilerates = df_compilerates[df_compilerates['motor dilution']==1.0]

df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
df_contractionspeed = df_contractionspeed[df_contractionspeed['motor dilution']==1.0]

df_comsol = pd.read_csv('../../analyzed_data/comsol_unitcell_compiled.csv', sep=',')

df_fits = pd.read_csv('../../analyzed_data/diffusion_fits.csv', sep=',')
df_fits = df_fits[(df_fits['ATP (uM)'] == 1400) & (df_fits['pluronic'] == 1)]
#%%
axis_fontsize = 30
label_fontsize = 48

motor_list = df_compilerates['motor'].unique()
x_pos = np.arange(0, len(motor_list), 1)

motor_speed = dict({'k401bac':250,'k401':600,
                    'ncd236':115, 'ncd281':90})
x_dict = dict({m:x for m,x in zip(motor_list,x_pos)})
motor_color = ['red', 'green', 'orange', 'blue']

color_dict = dict({'ncd281' : 'red',
                   'ncd236' : 'green',
                   'k401bac' : 'rebeccapurple', 
                   'k401': 'dodgerblue'})

fig, ax = plt.subplots(1,2,figsize=(9,4))
for motor, _d in df_compilerates.groupby('motor'):
    d = _d[(_d['ATP (uM)']==1400) 
            & (_d['motor dilution']==1.0)
            & (_d['pluronic']==1)]
    ax[0].errorbar(motor_speed[motor], d['mean_rate'].values[0],
                yerr=np.array(d['mean_rate'].values[0] - d['rate_low'].values[0],
                                d['rate_high'].values[0] - d['mean_rate'].values[0]),
                label=np.char.capitalize(motor), color=color_dict[motor], ms=8, marker='o',
                linestyle='None')

    _df_fits = df_fits[df_fits['motor']==motor]
    ax[1].errorbar(motor_speed[motor], _df_fits['D (median)'],
                   yerr=np.array(_df_fits['D (median)'] - _df_fits['D (1st quartile)'], 
                                 _df_fits['D (3rd quartile)'] - _df_fits['D (median)']), 
                                 ms=8, marker='o', linestyle='None', color=color_dict[motor],
                                 label=np.char.capitalize(motor))
#ax.xaxis.set_tick_params(fontsize=24)

for a in ax:
    a.set_xlabel('motor speed [nm/s]', fontsize=16)
    a.legend(loc=2, fontsize=12)
ax[0].set_ylabel('contraction rate [1/s]', fontsize=16)
ax[1].set_ylabel('effective diffusion\nconstant [µm$^2$/s]', fontsize=16)
ax[1].set_yticks([0, 0.005, 0.01, 0.015, 0.02])

ax[0].text(-95, 0.0183, '(A)', fontsize=20)
ax[1].text(-135, 0.02, '(B)', fontsize=20)

#ax.set_xlim([0, 150])
#ax.set_ylim([0, 0.0025])
fig.tight_layout()
plt.savefig('../../figures/FigX_contraction_rates_motorspeeds.pdf', bbox_inches='tight',
            facecolor='white')

# %%
