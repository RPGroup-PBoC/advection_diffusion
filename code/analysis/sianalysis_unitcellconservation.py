#%%
# Macroscopic deformations of fluorescent unit cells
import sys
sys.path.insert(0, '../')
import os
import active_matter_pkg as amp
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from multiprocess import Pool, cpu_count
tqdm.pandas()
amp.viz.plotting_style()

def analyze_photobleach_single(data):
    file, intensity_thresh = data
    n_start = 0
    n_tot = 20
    small_area = 800
    large_area = 4000

    # Include camera offset
    offset = 1920
    
    # Include threshold to number of layers added to preserve total intensity
    layer_thresh = 50

    df = amp.image_processing.analyze_photobleach(file, offset, n_tot, 
                                                    intensity_thresh, layer_thresh, 
                                                    n_start=n_start, new_thresholding=True,
                                                    small_area = small_area, large_area=large_area,
                                                    thresh_method='custom')
    df['threshold_fraction'] = intensity_thresh

    return df

def apply_parallel(filelist, func):
    with Pool(cpu_count()) as p:
        ret_list = list(tqdm(p.imap(func, filelist), total=len(filelist)))

    return pd.concat(ret_list)

def func_group_apply(df, param_list):
    return df.groupby(param_list).apply(analyze_photobleach_single)

filetxt = '../../analyzed_data/analyzing_filenames.txt'

with open(filetxt, 'r') as filestream:
    files = [line[:-1] for line in filestream if 'slide' in line]

root = '../../../data/active_stress/photobleach_data/'
filelist = [file for file in os.listdir(root) if file in files]

threshold_list = np.arange(0.99, 0.999, 0.001)

data = list(itertools.product(filelist, threshold_list))
# %%
if __name__ == '__main__':
    df_compiled2 = apply_parallel(data, analyze_photobleach_single)
# %%
df_compiled2.to_csv('../../analyzed_data/unitcell_threshsweep_2.csv',
                   sep=',')
# %%
df_compiled = pd.read_csv('../../analyzed_data/unitcell_threshsweep_2.csv',
                   sep=',')

# Compute normalized areas for each of the data
df_norm = pd.DataFrame()
for info,d in df_compiled.groupby(['filename','cellID','threshold_fraction']):
    if len(d) < 4:
        continue
    area_norm = d['area'] / d[d['time'] == d['time'].min()]['area'].values[0]
    times = d['time']
    abs_times = (d['time'] - 1) * d['time interval (s)'].values[0] + 2
    abs_times[abs_times < 0] = 0

    _df_norm = pd.DataFrame(np.array([area_norm, times, abs_times]).T,
                            columns=('area_normalized','time', 'absolute_time'))
    _df_norm['filename'] = info[0]
    _df_norm['cellID'] = info[1]
    _df_norm['threshold_fraction'] = info[2]

    df_norm = pd.concat([df_norm, _df_norm], ignore_index=True)
#%%
df_diffusion = pd.read_csv('../../analyzed_data/comsol_unitcell_compiled_v2.csv',
                           sep=',')
df_diffusion = df_diffusion[df_diffusion['alpha (1/s)']==0.002]
# %%
# Compile averages for each
df_avg = pd.DataFrame()
for info, df in df_norm.groupby(['threshold_fraction']):

    alpha = 0.002
    
    for time, d in df.groupby('time'):
        area_mean = np.quantile(d['area_normalized'], 0.5)

        _df_avg = pd.DataFrame([[info[0], d['absolute_time'].values[0], np.round(alpha, 4), 
                                    area_mean]],
                                columns = ('threshold', 'time', 'alpha (1/s)',
                                            'area_normalized (median)'))

        df_avg = pd.concat([df_avg, _df_avg], ignore_index=True)
# %%
def compute_residual(array1, array2):
    return np.dot(array1 - array2, array1 - array2)

def fit_diffusion(expt_array, df_comsol, alpha):
    # Perform residual for the no-diffusion case
    nodiff_array = np.array([(1 - t * alpha)**2 for t in df_comsol['time'].unique()])
    df_fits = pd.DataFrame([[0, compute_residual(expt_array, nodiff_array)]],
                           columns=('D (um^2/s)', 'residual'))

    for diff, _df_comsol in df_comsol.groupby('D (um^2/s)'):
        comsol_array = _df_comsol['area'].values[:] / _df_comsol['area'].values[0]
        residual = compute_residual(expt_array, comsol_array)
        _df_fits = pd.DataFrame([[diff, residual]], 
                                columns=('D (um^2/s)', 'residual'))
        df_fits = pd.concat([df_fits, _df_fits], ignore_index=True)

    return df_fits[df_fits['residual'] == df_fits['residual'].min()]['D (um^2/s)'].values[0]


df_diff = pd.DataFrame()
for alpha, df in df_avg.groupby('alpha (1/s)'):
    d_comsol = df_diffusion[df_diffusion['alpha (1/s)'] == alpha].drop_duplicates(subset=('D (um^2/s)', 'time'), keep='first')

    for info, d in df.groupby(['threshold']):

        absolute_times = d['time'].unique()[:10]
        _d_comsol = d_comsol[d_comsol['time'].isin(absolute_times)]
        _d = d[d['time'].isin(absolute_times)]
        
        # Compute for median
        diff_mean = fit_diffusion(_d['area_normalized (median)'].values[:],
                                _d_comsol, alpha)

        _df_diff = pd.DataFrame([[info[0], alpha, diff_mean]],
                                columns=('threshold', 'alpha (1/s)',
                                         'D (median)'))
        
        df_diff = pd.concat([df_diff, _df_diff], ignore_index = True)

#df_diff.to_csv('../../analyzed_data/diffusion_fits_SI_v2.csv', sep=',')
# %%
df_comsol = pd.read_csv('../../analyzed_data/comsol_unitcell_compiled_v2.csv', sep=',')

alpha = 0.0020 
#dt = df_truncated['time interval (s)'].values[0]

fig, ax = plt.subplots(1, 1, figsize=(6,5))

t_theor = np.linspace(0, 100, 1000)

d_comsol = df_comsol[df_comsol['alpha (1/s)']==alpha].drop_duplicates(subset=('alpha (1/s)', 'D (um^2/s)', 'time'))

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
    if diff > 0.003:
        continue
    ax.plot(_d['time'], _d['area'] / (_d[_d['time']==0]['area'].values[0]),
                color=dict_diff[diff], lw=3, zorder=-1)
ax.plot(t_theor, a_theor, ls='--', lw=3, label='pure contraction', color='dodgerblue', zorder=-2)

"""for time,d in df_truncated.groupby('time'):
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
                facecolor='white', marker='P')"""
        
ax.set_xlabel('time [s]', fontsize=16)
ax.set_ylabel('normalized area', fontsize=16)
#ax.legend(loc=2, ncols=3)
ax.set_title(r'$\alpha =$ %.4f sec$^{-1}$' %alpha)
test = ax.contourf([[np.nan, np.nan],[np.nan,np.nan]], levels = np.linspace(0.5,6,1000), cmap=newcmap)
#ax.scatter(np.nan, np.nan, edgecolor='tomato', s=50,
#            facecolor='white', marker='^', label='1st quartile (expt)')
#ax.scatter(np.nan, np.nan, edgecolor='tomato', s=50,
#            facecolor='white', marker='o', label='median (expt)')
#ax.scatter(np.nan, np.nan, edgecolor='tomato', s=50,
#            facecolor='white', marker='P', label='3rd quartile (expt)')
#plt.colorbar(test, ax=ax, label=r'$D$ ($\times 10^{-3}$ µm$^2$/s)', 
#            ticks=d_comsol['D (um^2/s)'].unique()*1000)
ax.set_ylim(0.6, 1.2)
ax.legend(loc=3)
# %%
fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.scatter((1 - df_diff['threshold']), df_diff['D (median)'],
           edgecolor='rebeccapurple', facecolor='white')
ax.set_xlabel('tolerance')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.set_xticklabels([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.set_ylabel('Fit $D_\mathrm{eff}$ [µm$^2$/s]')
plt.savefig('../../figures/SIFigX_thresholding.pdf',
            bbox_inches='tight', facecolor='white')
# %%
