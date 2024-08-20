#%%
import sys
sys.path.insert(0,'../')
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import active_matter_pkg as amp
amp.viz.plotting_style()
tqdm.pandas()

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x, pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = mticker.FuncFormatter(g)

#root = '../../../data/active_stress/photobleach_data'
#data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])

df_graticule = pd.read_csv('../../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
um_per_pxl = df_graticule['micron_per_pixel'].values[0]
activation_radius = 125 #µm
df_t2c = pd.read_csv('../../analyzed_data/time_to_contraction.csv', sep=',')

df_compiled = pd.read_csv('../../analyzed_data/unitcell_features_compiled.csv', sep=',')
df_truncated = df_compiled[(~df_compiled['filename'].str.contains('12-05')
                        & (~df_compiled['filename'].str.contains('11-30')))]
df_truncated = df_truncated[(df_truncated['motor']=='ncd236')
                            & (df_truncated['ATP (uM)']==1400)
                            & (df_truncated['pluronic']==1)
                            & (df_truncated['motor dilution']==1.0)]

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_multithreading.csv', sep=',')
df_compilerates = df_compilerates[df_compilerates['motor dilution']==1.0]
df_compile = df_compilerates[(df_compilerates['motor']=='ncd236')
                                & (df_compilerates['motor dilution']==1.0)
                                & (df_compilerates['ATP (uM)']==1400)
                                & (df_compilerates['pluronic']==1)]


df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
df_contractionspeed = df_contractionspeed[df_contractionspeed['motor dilution']==1.0]

df_comsol = pd.read_csv('../../analyzed_data/comsol_unitcell_compiled.csv', sep=',')
#%%  
alpha = 0.0020 
dt = df_truncated['time interval (s)'].values[0]

fig, ax = plt.subplots(1, 1, figsize=(6,5))

t_theor = np.linspace(0, 100, 1000)

d_comsol = df_comsol[df_comsol['alpha (1/s)']==alpha]

a_theor = (1 - alpha * t_theor)**2

n = np.argwhere(d_comsol.sort_values('alpha (1/s)', ascending=True)['alpha (1/s)'].unique()==alpha)[0][0]

color = matplotlib.colormaps['Reds']
x = np.linspace(0.4, 1.0, len(d_comsol['D (um^2/s)'].unique()))
colors = [color(_x) for _x in x]

x_cont = np.linspace(0.4, 1.0, 1000)
newcmap = matplotlib.colors.ListedColormap([color(_x) for _x in x_cont])
difflist = np.sort(d_comsol['D (um^2/s)'].unique())
dict_diff = dict(zip(difflist,colors))

for diff,_d in d_comsol.groupby('D (um^2/s)'):
    if diff > 0.006:
        continue
    ax.plot(_d['time'], _d['area'] / (_d[_d['time']==0]['area'].values[0]),
                color=dict_diff[diff], lw=3, zorder=-1)
ax.plot(t_theor, a_theor, ls='--', lw=3, label='pure contraction', color='dodgerblue', zorder=-2)

for time,d in df_truncated.groupby('time'):
    if time == 0:
        ax.scatter(time*dt,np.quantile(d['area_normalized'], 0.25), 
                    edgecolor='tomato', s=50,
                    facecolor='white', marker='^')
        ax.scatter(time*dt,d['area_normalized'].median(), 
                edgecolor='tomato', s=50,
                facecolor='white', marker='o')
        ax.scatter(time*dt,np.quantile(d['area_normalized'], 0.75), 
                edgecolor='tomato', s=50,
                facecolor='white', marker='P')
    elif time >= 10:
        continue
    else:
        ax.scatter((time - 1) * dt + 2, np.quantile(d['area_normalized'], 0.25), 
                    edgecolor='tomato', s=50,
                    facecolor='white', marker='^')
        ax.scatter((time - 1) * dt + 2, d['area_normalized'].median(), 
                edgecolor='tomato', s=50,
                facecolor='white', marker='o')
        ax.scatter((time - 1) * dt + 2, np.quantile(d['area_normalized'], 0.75), 
                edgecolor='tomato', s=50,
                facecolor='white', marker='P')
        
ax.set_xlabel('time [s]', fontsize=16)
ax.set_ylabel('normalized area', fontsize=16)
#ax.legend(loc=2, ncols=3)
ax.set_title(r'$\alpha =$ %.4f sec$^{-1}$' %alpha)
test = ax.contourf([[np.nan, np.nan],[np.nan,np.nan]], levels = np.linspace(0.5,6,1000), cmap=newcmap)
ax.scatter(np.nan, np.nan, edgecolor='tomato', s=50,
            facecolor='white', marker='^', label='1st quartile (expt)')
ax.scatter(np.nan, np.nan, edgecolor='tomato', s=50,
            facecolor='white', marker='o', label='median (expt)')
ax.scatter(np.nan, np.nan, edgecolor='tomato', s=50,
            facecolor='white', marker='P', label='3rd quartile (expt)')
plt.colorbar(test, ax=ax, label=r'$D$ ($\times 10^{-3}$ µm$^2$/s)', 
            ticks=d_comsol['D (um^2/s)'].unique()*1000)
ax.set_ylim(0.6, 1.2)
ax.legend(loc=3)

fig.tight_layout()
plt.savefig('../../figures/area_trajectory_ncd236.pdf',
            bbox_inches='tight', facecolor='white')


# %%
