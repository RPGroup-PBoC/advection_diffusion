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
# %%
# Perform fit
def hill_fn(x, a, k):
    return a * (x / k) / (1 + (x / k))

def ncd_fn(x, kdet, kD):
    return (-3.8 * kdet**2 * kD * x + 5.2 * (kdet * (kD * x)**2)) / (kdet**2 + kdet * kD * x + (kD * x)**2)

motor_list = ['ncd236', 'k401bac']
fig, ax = plt.subplots(2,2,figsize=(19,16))

for m in range(2):
    motor = motor_list[m]
    marker = 'D'
    if motor == 'k401bac':
        color = 'rebeccapurple'
    elif motor == 'ncd236':
        color = 'green'

    atp_array = np.zeros((4, len(df_compilerates[(df_compilerates['motor']==motor) & (df_compilerates['pluronic']==1)]['ATP (uM)'].unique())))
    n=0
    for atp,_d in df_compilerates[(df_compilerates['motor']==motor) & (df_compilerates['pluronic']==1)].groupby('ATP (uM)'):
        if motor == 'ncd236':
            if (atp == 300) or (atp == 75):
                continue
        elif motor == 'k401bac':
            if (atp == 300):
                continue
        atp_array[0,n] = atp
        atp_array[1,n] = _d['mean_rate'].values[0]
        atp_array[2,n] = _d['mean_rate'].values[0] - _d['rate_low'].values[0]
        atp_array[3,n] = _d['rate_high'].values[0] - _d['mean_rate'].values[0]
        n+=1

    popt, pcov = curve_fit(hill_fn, atp_array[0,:], atp_array[1,:], p0=np.array([atp_array[1,-1], 25]))
    a_opt, k_opt = popt

    x_theor = np.linspace(3, 3000, 10000)
    y_theor = hill_fn(x_theor, a_opt, k_opt)
    y_min = hill_fn(x_theor, a_opt - pcov[0,0]**0.5, k_opt + pcov[1,1]**0.5)
    y_max = hill_fn(x_theor, a_opt + pcov[0,0]**0.5, k_opt - pcov[1,1]**0.5)

    for atp,_d in df_compilerates[(df_compilerates['motor']==motor) & (df_compilerates['pluronic']==1)].groupby('ATP (uM)'):
        if motor == 'ncd236':
            if (atp == 300) or (atp == 75):
                continue
        elif motor == 'k401bac':
            if (atp == 300):
                continue
        ax[m,0].errorbar(atp, _d['mean_rate'].values[0], 
                    yerr=np.array(_d['mean_rate'].values[0] - _d['rate_low'].values[0],
                                _d['rate_high'].values[0] - _d['mean_rate'].values[0]),
                    color=color, marker=marker, ms=11, lw=3)
    ax[m,0].plot(x_theor, y_theor, color='black', label=r'$K_\mathrm{ATP}$ = %.1d $\pm$ %.1d µM' %(k_opt, pcov[1,1]**0.5))
    ax[m,0].fill_between(x_theor, y_min, y_max, color='gray', alpha=0.4, label=r'$\pm \sigma$ in $K_{D}$ and amplitude')
    ax[m,0].errorbar(np.nan, np.nan, yerr=1, color=color, ms=11, lw=3,
                     linestyle='', marker=marker, label='expt')
    if motor == 'ncd236':
        _df_fits = df_fits[df_fits['motor']=='ncd236'].sort_values('ATP (uM)')
    else:
        _df_fits = df_fits[df_fits['motor']=='k401bac'].sort_values('ATP (uM)')

    popt, pcov = curve_fit(hill_fn, _df_fits['ATP (uM)'], _df_fits['D (median)'], 
                           p0=np.array([_df_fits['ATP (uM)'].max(), 25]))
    d_opt, kd_opt = popt
    d_theor = hill_fn(x_theor, d_opt, kd_opt)

    amplitude_min = d_opt - pcov[0,0]**0.5
    if amplitude_min < 0:
        amplitude_min = 0

    kd_min = kd_opt - pcov[1,1]**0.5
    if kd_min < 0:
        kd_min = 0
        
    #d_min = hill_fn(x_theor, amplitude_min, kd_opt + pcov[1,1]**0.5)
    #d_max = hill_fn(x_theor, d_opt + pcov[0,0]**0.5, kd_min)
  
    ax[m,1].scatter(_df_fits['ATP (uM)'], _df_fits['D (median)'], 
                    facecolor='white', edgecolor=color, 
                    marker=marker, s=120, linestyle='None',
                    label='median')
    ax[m,1].errorbar(_df_fits['ATP (uM)'], _df_fits['D (median)'], 
                     yerr=np.array(_df_fits['D (median)'] - _df_fits['D (1st quartile)'], 
                         _df_fits['D (median)'] - _df_fits['D (3rd quartile)']), 
                        color=color, zorder=-5, marker=marker, ms=11, lw=3,
                        label='±25% of distribution', linestyle='')
    #ax[m,1].plot(x_theor, d_theor, color='black', label=r'$K_\mathrm{ATP}$ = %.1d $\pm$ %.1d µM' %(kd_opt, pcov[1,1]**0.5))
    #ax[m,1].fill_between(x_theor, d_min, d_max, 
    #                     color='gray', alpha=0.4, 
    #                     label=r'$\pm \sigma$ in $K_{D}$ and amplitude')
    ax[m,0].set_xlim([3, 3000])
    ax[m,1].set_ylabel('effective diffusion\nconstant [µm$^2$/s]', fontsize=28)
ax[0,1].set_yticks([0, 0.0005, 0.001, 0.0015, 0.002])

ax[0,0].set_ylabel('Ncd236\ncontraction rate [1/s]', fontsize=28)
ax[1,0].set_ylabel('K401 (bacterial)\ncontraction rate [1/s]', fontsize=28)

ax[0,0].set_ylim([0, 0.005])
ax[1,0].set_ylim([0, 0.009])
ax[1,1].set_ylim([-0.0002, 0.0052])

for a in ax.ravel():
    a.legend(loc=2, fontsize=22)
    a.set_xscale('log')
    a.set_xlabel('ATP concentration [µM]', fontsize=28)
    a.tick_params(axis='both', which='major', labelsize=28)

ax[0,0].text(1, 0.005, '(A)', va='center', ha='right', fontsize=36)
ax[0,1].text(3.6, 0.002, '(B)', va='center', ha='right', fontsize=36)
ax[1,0].text(1, 0.0081, '(C)', va='center', ha='right', fontsize=36)
ax[1,1].text(6.5, 0.0052, '(D)', va='center', ha='right', fontsize=36)
fig.tight_layout()
plt.savefig('../../figures/FigX_ATP_speeds.pdf', bbox_inches='tight',
            facecolor='white')

# %%
