#%%
# Compiling figure from data analysis of unit cells
import os
import sys
sys.path.insert(0,'../')
import active_matter_pkg as amp
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
amp.viz.plotting_style()

df_compiled = pd.read_csv('../../analyzed_data/unitcell_features_compiled.csv', sep=',')
df_truncated = df_compiled[(~df_compiled['filename'].str.contains('12-05')
                        & (~df_compiled['filename'].str.contains('11-30')))]
#df_truncated = df_truncated[(df_truncated['ATP (uM)']==1400) & (df_truncated['pluronic']==1)]
df_truncated = df_truncated.loc[:, ~df_truncated.columns.str.contains('^Unnamed')]

df_compilerates = pd.read_csv('../../analyzed_data/compiledrates_allmotors_multithreading.csv', sep=',')
df_compilerates = df_compilerates[df_compilerates['motor dilution']==1.0]

df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
df_contractionspeed = df_contractionspeed[df_contractionspeed['motor dilution']==1.0]

df_comsol = pd.read_csv('../../analyzed_data/comsol_unitcell_compiled.csv', sep=',')
# %%
# Compile averages for each
df_avg = pd.DataFrame()
for info, df in df_truncated.groupby(['motor', 'ATP (uM)', 'pluronic']):
    mot, atp, plur = info

    if (mot == 'ncd236') and ((atp == 300) or (atp == 75)):
        continue

    if len(df['time interval (s)'].unique()) > 1:

        # Return most abundant dataset if there is discrepancy in time interval length
        interval = df['time interval (s)'].value_counts().index[0]
        df = df[df['time interval (s)'] == interval]
    
    else:
        interval = df['time interval (s)'].values[0]

    alpha = df_compilerates[(df_compilerates['motor'] == mot)
                            & (df_compilerates['ATP (uM)'] == atp)
                            & (df_compilerates['pluronic'] == plur)]['mean_rate'].values[0]
    alpha_low = df_compilerates[(df_compilerates['motor'] == mot)
                            & (df_compilerates['ATP (uM)'] == atp)
                            & (df_compilerates['pluronic'] == plur)]['rate_low'].values[0]
    alpha_high = df_compilerates[(df_compilerates['motor'] == mot)
                            & (df_compilerates['ATP (uM)'] == atp)
                            & (df_compilerates['pluronic'] == plur)]['rate_high'].values[0]
    
    for time, d in df.groupby('time'):
        area_mean = np.quantile(d['area_normalized'], 0.5)
        area_low = np.quantile(d['area_normalized'], 0.25)
        area_high = np.quantile(d['area_normalized'], 0.75)

        if atp == 12.5:
            _df_avg = pd.DataFrame([[atp, plur, d['absolute_time'].values[0], np.round(alpha, 5), 
                                    np.round(alpha_low, 5), np.round(alpha_high, 5),
                                    area_low, area_mean, area_high]],
                                columns = ('ATP (uM)', 'pluronic', 'time', 'alpha (1/s)',
                                            'alpha_low', 'alpha_high',
                                            'area_normalized (1st quartile)', 
                                            'area_normalized (median)',
                                            'area_normalized (3rd quartile)'))
        else:
            _df_avg = pd.DataFrame([[atp, plur, d['absolute_time'].values[0], np.round(alpha, 4), 
                                    np.round(alpha_low, 4), np.round(alpha_high, 4),
                                    area_low, area_mean, area_high]],
                                columns = ('ATP (uM)', 'pluronic', 'time', 'alpha (1/s)',
                                            'alpha_low', 'alpha_high',
                                            'area_normalized (1st quartile)', 
                                            'area_normalized (median)',
                                            'area_normalized (3rd quartile)'))
        _df_avg['motor'] = mot

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
    d_comsol = df_comsol[df_comsol['alpha (1/s)'] == alpha].drop_duplicates(subset=('D (um^2/s)', 'time'), keep='first')
    alpha_low = df['alpha_low'].values[0]
    alpha_high = df['alpha_high'].values[0]

    for info, d in df.groupby(['motor', 'ATP (uM)', 'pluronic']):
        mot, atp, plur = info

        absolute_times = d['time'].unique()[:10]
        _d_comsol = d_comsol[d_comsol['time'].isin(absolute_times)]
        _d = d[d['time'].isin(absolute_times)]
        
        # Compute for median
        diff_mean = fit_diffusion(_d['area_normalized (median)'].values[:],
                                _d_comsol, alpha)

        # Compute for 1st quartile
        diff_low = fit_diffusion(_d['area_normalized (1st quartile)'].values[:],
                                _d_comsol, alpha)
        
        # Compute for 3rd quartile
        diff_high = fit_diffusion(_d['area_normalized (3rd quartile)'].values[:],
                                _d_comsol, alpha)
        
        _df_diff = pd.DataFrame([[atp, plur, alpha, alpha_low, alpha_high, diff_mean, diff_low, diff_high]],
                                columns=('ATP (uM)', 'pluronic', 'alpha (1/s)',
                                        'alpha_low', 'alpha_high',
                                         'D (median)', 'D (1st quartile)', 
                                         'D (3rd quartile)'))
        _df_diff['motor'] = mot
        
        df_diff = pd.concat([df_diff, _df_diff], ignore_index = True)

df_diff.insert(0, 'motor', df_diff.pop('motor'))
df_diff.to_csv('../../analyzed_data/diffusion_fits.csv', sep=',')
# %%
