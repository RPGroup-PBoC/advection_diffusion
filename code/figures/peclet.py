#%%
import sys
sys.path.insert(0, '../')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import active_matter_pkg as amp
from scipy.optimize import curve_fit

amp.viz.plotting_style()

df_fits = pd.read_csv('../../analyzed_data/diffusion_fits.csv', sep=',')
df_fits = df_fits[df_fits['pluronic']==1.0]
# %%
def linear_fn(x, a, b):
    return a * x + b

x_theor = np.linspace(-0.001, 0.025, 100)

# Define natural length scale L to be microtubule length
L = 1.5

df_species = df_fits[(df_fits['ATP (uM)']==1400.0)]
dict_motor = dict({'ncd281' : 'red',
                   'ncd236' : 'green',
                   'k401bac' : 'rebeccapurple', 
                   'k401': 'dodgerblue'})

df_ncdatp = df_fits[(df_fits['motor']=='ncd236')]
df_k401atp = df_fits[(df_fits['motor']=='k401bac')]

popt_50, pcov_50 = curve_fit(linear_fn, df_fits['D (median)'], df_fits['alpha (1/s)'] * L**2, 
                       p0=np.array([df_fits['D (median)'].max() / (L**2 * df_fits['alpha (1/s)'].max()), 0]))
a_50, b_50 = popt_50
y_50 = linear_fn(x_theor, a_50, b_50)

fig, ax = plt.subplots(1,1,figsize=(6,6))

ax.errorbar(df_ncdatp['D (median)'], df_ncdatp['alpha (1/s)'] * L**2, 
                yerr= L**2 * np.array(df_ncdatp['alpha (1/s)'] - df_ncdatp['alpha_low'], 
                        df_ncdatp['alpha_high'] - df_ncdatp['alpha (1/s)']), 
                color='green', zorder=-5, marker='D', ms=6, lw=3,
                linestyle='', label='Ncd236 ATP')

ax.errorbar(df_k401atp['D (median)'], df_k401atp['alpha (1/s)'] * L**2, 
                yerr=L**2 * np.array(df_k401atp['alpha (1/s)'] - df_k401atp['alpha_low'], 
                        df_k401atp['alpha_high'] - df_k401atp['alpha (1/s)']), 
                color='rebeccapurple', zorder=-5, marker='D', ms=6, lw=3,
                linestyle='', label='K401bac ATP')

for m,d in df_species.groupby('motor'):
        ax.errorbar(d['D (median)'], d['alpha (1/s)'] * L**2, 
                        yerr= L**2 * np.array(d['alpha (1/s)'] - d['alpha_low'], 
                                d['alpha_high'] - d['alpha (1/s)']), 
                        markeredgecolor=dict_motor[m], 
                        markerfacecolor='white', c=dict_motor[m],
                        zorder=-5, marker='o', ms=6, lw=3,
                        linestyle='', label='%s (Fig 4)' %np.char.capitalize(m))

ax.plot(x_theor, y_50, color='black', lw=2, zorder=-6,
        label='$\mathrm{Pe}_\mathrm{med}$ = %.1f ± %.1f' %(a_50, np.sqrt(pcov_50[0,0])))

ax.set_ylabel(r'$\alpha \, L_\mathrm{char}^2$ [µm$^2$/s]', fontsize=16)
ax.set_xlabel(r'$D_\mathrm{eff}$ [µm$^2$/s]', fontsize=16)
ax.legend(loc=4, fontsize=11, ncol=2)
ax.set_xlim(-0.0005, 0.017)
ax.set_ylim(-0.0005, 0.040)
plt.savefig('../../figures/FigX_peclet.pdf',
            bbox_inches='tight', facecolor='white')
# %%
# Define natural length scale L to be microtubule length
L = 1.5

popt_25, pcov_25 = curve_fit(linear_fn, df_fits['D (1st quartile)'], df_fits['alpha (1/s)'] * L**2,
                       p0=np.array([df_fits['D (1st quartile)'].max() / (L**2 * df_fits['alpha (1/s)'].max()), 0]))
a_25, b_25 = popt_25
y_25 = linear_fn(x_theor, a_25, b_25)

popt_50, pcov_50 = curve_fit(linear_fn, df_fits['D (median)'], df_fits['alpha (1/s)'] * L**2, 
                       p0=np.array([df_fits['D (median)'].max() / (L**2 * df_fits['alpha (1/s)'].max()), 0]))
a_50, b_50 = popt_50
y_50 = linear_fn(x_theor, a_50, b_50)

popt_75, pcov_75 = curve_fit(linear_fn, df_fits['D (3rd quartile)'], df_fits['alpha (1/s)'] * L**2,
                       p0=np.array([df_fits['D (3rd quartile)'].max() / (L**2 * df_fits['alpha (1/s)'].max()), 0]))
a_75, b_75 = popt_75
y_75 = linear_fn(x_theor, a_75, b_75)

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.errorbar(df_fits['D (1st quartile)'], df_fits['alpha (1/s)'] * L**2, 
                yerr=L**2 * np.array(df_fits['alpha (1/s)'] - df_fits['alpha_low'], 
                        df_fits['alpha_high'] - df_fits['alpha (1/s)']), 
                color='tomato', zorder=-5, marker='^', ms=6, lw=3,
                linestyle='', label='1st quartile')

ax.errorbar(df_fits['D (median)'], df_fits['alpha (1/s)'] * L**2, 
                yerr=L**2 * np.array(df_fits['alpha (1/s)'] - df_fits['alpha_low'], 
                        df_fits['alpha_high'] - df_fits['alpha (1/s)']), 
                color='dodgerblue', zorder=-5, marker='o', ms=6, lw=3,
                linestyle='', label='median')

ax.errorbar(df_fits['D (3rd quartile)'], df_fits['alpha (1/s)'] * L**2, 
                yerr=L**2 * np.array(df_fits['alpha (1/s)'] - df_fits['alpha_low'], 
                        df_fits['alpha_high'] - df_fits['alpha (1/s)']), 
                color='rebeccapurple', zorder=-5, marker='P', ms=6, lw=3,
                linestyle='', label='3rd quartile')

ax.plot(x_theor, y_25, color='tomato', lw=2, 
        label='$\mathrm{Pe}_\mathrm{25}$ = %.1f ± %.1f' %(a_25, np.sqrt(pcov_25[0,0])))

ax.plot(x_theor, y_50, color='dodgerblue', lw=2, 
        label='$\mathrm{Pe}_\mathrm{med}$ = %.1f ± %.1f' %(a_50, np.sqrt(pcov_50[0,0])))

ax.plot(x_theor, y_75, color='rebeccapurple', 
        lw=2, label='$\mathrm{Pe}_\mathrm{75}$ = %.1f ± %.1f' %(a_75, np.sqrt(pcov_75[0,0])))

ax.set_ylabel(r'$\alpha \, L_\mathrm{char}^2$ [µm$^2$/s]', fontsize=16)
ax.set_xlabel(r'$D_\mathrm{eff}$ [µm$^2$/s]', fontsize=16)
ax.legend(loc=4, fontsize=11, ncol=2)
ax.set_xlim(-0.0005, 0.017)
ax.set_ylim(-0.0005, 0.040)
plt.savefig('../../figures/SIFigX_peclet.pdf',
            bbox_inches='tight', facecolor='white')
# %%
