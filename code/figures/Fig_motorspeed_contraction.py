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

df_compiled = pd.read_csv('../../analyzed_data/unitcell_features_compiled.csv', sep=',')
df_truncated = df_compiled[(~df_compiled['filename'].str.contains('12-05')
                        & (~df_compiled['filename'].str.contains('11-30')))]

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_multithreading.csv', sep=',')
df_compilerates = df_compilerates[df_compilerates['motor dilution']==1.0]


df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
df_contractionspeed = df_contractionspeed[df_contractionspeed['motor dilution']==1.0]

df_comsol = pd.read_csv('../../analyzed_data/comsol_unitcell_compiled.csv', sep=',')

df_fits = pd.read_csv('../../analyzed_data/diffusion_fits.csv', sep=',')
df_fits = df_fits[df_fits['pluronic']==1]

df_parameters = pd.read_csv('../../analyzed_data/atp_speed_parameters.csv', sep=',')

#%%
def hill_fn(x):
    a, k = df_parameters[df_parameters['motor']=='k401'][['alpha', 'k_ATP']].values[0]

    return a * (x / k) / (1 + (x / k))

def ncd_fn(x):
    kdet, kD = df_parameters[df_parameters['motor']=='ncd'][['k_detach','k_ATP']].values[0]

    return (-3.8 * kdet**2 * kD * x + 5.2 * (kdet * (kD * x)**2)) / (kdet**2 + kdet * kD * x + (kD * x)**2)

motor_list = ['ncd236', 'k401bac']

fig, ax = plt.subplots(1,2,figsize=(19,8))

for m in range(2):
    motor = motor_list[m]

    for atp,_d in df_compilerates[(df_compilerates['motor']==motor) & (df_compilerates['pluronic']==1)].groupby('ATP (uM)'):

        if motor == 'ncd236':
            if (atp == 300) or (atp == 75):
                continue
            else:
                speed = ncd_fn(atp)
                ax[0].errorbar(speed, _d['mean_rate'].values[0], 
                    yerr=np.array(_d['mean_rate'].values[0] - _d['rate_low'].values[0],
                                _d['rate_high'].values[0] - _d['mean_rate'].values[0]),
                    color='dodgerblue', marker='o', ms=10)
        elif motor == 'k401bac':
            if (atp == 300):
                continue
            else:
                speed = hill_fn(atp)
                ax[0].errorbar(speed, _d['mean_rate'].values[0], 
                    yerr=np.array(_d['mean_rate'].values[0] - _d['rate_low'].values[0],
                                _d['rate_high'].values[0] - _d['mean_rate'].values[0]),
                    color='rebeccapurple', marker='o', ms=10)

    if motor == 'ncd236':
        _df_fits = df_fits[df_fits['motor']=='ncd236']
        ax[1].scatter(ncd_fn(_df_fits['ATP (uM)']), _df_fits['D (median)'], 
                        color='dodgerblue', marker='o', s=100, linestyle='None',
                        label='Ncd236 ATP')
    else:
        _df_fits = df_fits[df_fits['motor']=='k401bac']
        ax[1].scatter(hill_fn(_df_fits['ATP (uM)']), _df_fits['D (median)'], 
                        color='rebeccapurple', marker='o', s=100, linestyle='None',
                        label='K401 ATP')
ax[0].scatter(np.nan, np.nan, s=100, color='dodgerblue', label='Ncd236 ATP')
ax[0].scatter(np.nan, np.nan, s=100, color='rebeccapurple', label='K401 ATP')

motor_list = df_compilerates['motor'].unique()
x_pos = np.arange(0, len(motor_list), 1)
motor_speed = dict({'k401bac':250,'k401':600,
                    'ncd236':115, 'ncd281':90})

for motor, _d in df_compilerates.groupby('motor'):
    d = _d[(_d['ATP (uM)']==1400) 
            & (_d['motor dilution']==1.0)
            & (_d['pluronic']==1)]
    ax[0].scatter(motor_speed[motor], d['mean_rate'].values[0], 
                  s=100, alpha=1.0, color='green', zorder=10)
    ax[0].errorbar(motor_speed[motor], d['mean_rate'].values[0],
                yerr=np.array(d['mean_rate'].values[0] - d['rate_low'].values[0],
                                d['rate_high'].values[0] - d['mean_rate'].values[0]),
                color='green', zorder=10)

    _df_fits = df_fits[(df_fits['motor']==motor) & (df_fits['ATP (uM)']==1400)]
    ax[1].scatter(motor_speed[motor], _df_fits['D (median)'], 
                marker='o', s=100, linestyle='None', color='green')

ax[0].scatter(np.nan, np.nan, color='green', marker='o', s=100, label='motor species')
ax[1].scatter(np.nan, np.nan, color='green', marker='o', s=100, label='motor species')

ax[0].set_ylabel('contraction rate [1/s]', fontsize=28)
ax[1].set_ylabel('effective diffusion\nconstant [µm$^2$/s]', fontsize=28)

#ax[0].set_ylim([0, 0.012])
#ax[0].set_xlim([-1, 1000])

for a in ax.ravel():
    a.legend(loc=2, fontsize=22)
    a.set_xlabel('effective motor speed [nm/s]', fontsize=28)
    a.tick_params(axis='both', which='major', labelsize=28)

#ax[0].text(1, 0.005, '(A)', va='center', ha='right', fontsize=36)
#ax[1].text(3.6, 0.003, '(B)', va='center', ha='right', fontsize=36)
ax[0].text(-70, 0.0190, '(A)', va='center', ha='right', fontsize=36)
ax[1].text(-70, 0.0141, '(B)', va='center', ha='right', fontsize=36)
fig.tight_layout()
plt.savefig('../../figures/FigX_ATPeffectivespeed_speeds.pdf', bbox_inches='tight',
            facecolor='white')
# %%
