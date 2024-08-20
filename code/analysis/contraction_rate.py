#%%
# Compiling figure from data analysis of unit cells
import sys
sys.path.insert(0, '../')
import active_matter_pkg as amp
from scipy.stats import linregress
import numpy as np
import arviz
import cmdstanpy
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
amp.viz.plotting_style()

df_compiled = pd.read_csv('../../analyzed_data/unitcell_features_allmotors_multithreaded_clean.csv', sep=',')

codepath = './contraction2.stan'
contraction_model = cmdstanpy.CmdStanModel(stan_file=codepath)
#%%
# Truncate data to only unmerged unit cells
df_truncated = pd.DataFrame()
for fid,_d in df_compiled.groupby(['filename','cellID']):
    f,id = fid
    try:
        d_lastmerge = _d[:_d[_d['merged']==1.0].index[0]-1]
    except:
        d = _d
    else:
        d = d_lastmerge[d_lastmerge['merged']==0.0]
    if (len(d) < 4) or (d['time'].min()>3):
        continue
    d['area_normalized'] = d['area']/d[d['time']==d['time'].min()]['area'].values[0]
    d['absolute_time'] = (d['time'] - 1) * d['time interval (s)'] + 2
    d.loc[d['absolute_time'] < 0,'absolute_time'] = 0
    if '11-30' in f:
        d['motor dilution'] = 0.8
    elif '12-05' in f:
        d['motor dilution'] = 0.9
    else:
        d['motor dilution'] = 1.0
    df_truncated = pd.concat([df_truncated,d],ignore_index=True)
df_truncated.to_csv('../../analyzed_data/unitcell_features_compiled.csv',
                    sep=',')
df_contractionspeed = pd.DataFrame()
for dataparams,d in df_compiled.groupby(['filename','motor', 'ATP (uM)', 'pluronic']):
    _root, mot, atp, pluronic = dataparams
    #if (atp > 30) and (atp < 300):
    #    if ('02-14' not in _root) and ('02-15' not in _root):
    #        continue
    for id, _d_ in d.groupby('cellID'):
        if (len(_d_) < 4) or (_d_['time'].min()>3):
            continue
        # Find slope of contraction speed
        slope, intercept, r_value, p_value, std_err = linregress(_d_['time']*_d_['time interval (s)'], _d_['radius'])
        _d = pd.DataFrame([[id,np.abs(slope)]], columns=['cellID','contraction speed'])
        _d['filename'] = _root
        _d['num_pb'] = d['num_pb'].values[0]
        _d['radius'] = _d_['radius'].values[0]
        _d['motor'] = mot
        _d['ATP (uM)'] = atp
        _d['pluronic'] = pluronic
        if '11-30' in f:
            _d['motor dilution'] = 0.8
        elif '12-05' in f:
            _d['motor dilution'] = 0.9
        else:
            _d['motor dilution'] = 1.0
        df_contractionspeed = pd.concat([df_contractionspeed,_d], ignore_index=True)
df_contractionspeed.to_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
#df_contractionspeed.to_csv('../../analyzed_data/contractionspeed_pluronic.csv', sep=',')
#%%
df_contractionspeed = pd.read_csv('../../analyzed_data/contractionspeed_allmotors_multithreading.csv', sep=',')
#df_contractionspeed = df_contractionspeed[(df_contractionspeed['filename'].str.contains('02-2'))
#                                          & (df_contractionspeed['filename'].str.contains('2023'))
#                                          & (df_contractionspeed['motor']=='k401bac')]
num_steps = 100
df_compilerates = pd.DataFrame([])
for parameters,d in df_contractionspeed.groupby(['motor', 'ATP (uM)', 'pluronic']):
    motor, atp, pluronic = parameters
    d = d[(~d['filename'].str.contains('11-30')) & (~d['filename'].str.contains('12-05'))]
    if motor=='ncd236':
        df_t2c = pd.read_csv('../../analyzed_data/time_to_contraction.csv', sep=',')
    else:
        df_t2c = pd.read_csv('../../analyzed_data/time_to_contraction_%s.csv' %motor, sep=',')

    dataset = dict(N=len(d), x=d['radius'].values,
                    y=d['contraction speed'].values,
                    slope_mean=0, slope_std=1,
                    sigma_slope_mean=0, sigma_slope_std=1,
                    offset_mean=0.0, offset_std=1)
    samples = contraction_model.sample(data=dataset)
    sample_rewrite = arviz.from_cmdstanpy(posterior=samples)
    df_samples = sample_rewrite.to_dataframe()
    
    contraction_cred = np.zeros((4, num_steps))
    contraction_cred[0,:] = np.linspace(0, df_compiled[df_compiled['motor']==motor]['radius'].max() * 1.01, num_steps)

    for n_r in tqdm(range(num_steps)):
        contraction_sim = []
        for m,sig,b in df_samples[[('posterior','slope'),('posterior','sigma_slope'),('posterior','offset')]].values:
            contraction_sim.append(m*contraction_cred[0,n_r] + b)
        contraction_cred[1:3,n_r] = amp.stats.hpd(contraction_sim, 0.95)
        contraction_cred[3,n_r] = np.median(contraction_sim)
    _df = pd.DataFrame(contraction_cred.T, columns=['distance','cred_low','cred_high','median'])

    _df['mean_rate'] = df_samples[('posterior','slope')].mean()
    hpd_95 = amp.stats.hpd(df_samples[('posterior','slope')],0.95)
    _df['rate_low'] = hpd_95[0]
    _df['rate_high'] = hpd_95[1]
    _df['ATP (uM)'] = atp
    _df['pluronic'] = pluronic
    _df['motor'] = motor
    _df['motor dilution'] = 1.0
    df_compilerates = pd.concat([df_compilerates,_df],ignore_index=True)
df_compilerates.to_csv('../../analyzed_data/compiledrates_allmotors_multithreading.csv', sep=',')
#df_compilerates.to_csv('../../analyzed_data/compiledrates_pluronic.csv', sep=',')
# %%
from multiprocess import Pool, cpu_count

def separate_replicates(df):
    num_steps = 100

    motor, f, a, p = df[['motor','filename','ATP (uM)', 'pluronic']].values[0]

    m, _, _, _, _ = linregress(d['radius'], d['contraction speed'])

    dataset = dict(N=len(df), x=df['radius'].values,
                    y=df['contraction speed'].values,
                    slope_mean=0, slope_std=1,
                    sigma_slope_mean=0, sigma_slope_std=1,
                    offset_mean=0.0, offset_std=1)
    samples = contraction_model.sample(data=dataset)
    sample_rewrite = arviz.from_cmdstanpy(posterior=samples)
    df_samples = sample_rewrite.to_dataframe()
    
    contraction_cred = np.zeros((4, num_steps))
    contraction_cred[0,:] = np.linspace(0,df_compiled[df_compiled['motor']==motor]['radius'].max()*1.01,num_steps)

    for n_r in range(num_steps):
        contraction_sim = []
        for m,sig,b in zip(df_samples[('posterior','slope')].values, df_samples[('posterior','sigma_slope')].values, df_samples[('posterior','offset')].values):
            contraction_sim.extend(np.random.normal(m*contraction_cred[0,n_r] + b, sig, size=len(d)))
        contraction_cred[1:3,n_r] = amp.stats.hpd(contraction_sim, 0.95)
        contraction_cred[3,n_r] = np.median(contraction_sim)
    _df = pd.DataFrame(contraction_cred.T, columns=['distance','cred_low','cred_high','mean'])

    _df['mean_rate'] = df_samples[('posterior','slope')].mean()
    hpd_95 = amp.stats.hpd(df_samples[('posterior','slope')],0.95)
    _df['rate_low'] = hpd_95[0]
    _df['rate_high'] = hpd_95[1]
    _df['ATP (uM)'] = a
    _df['pluronic'] = p
    _df['filename'] = f
    _df['num_pb'] = d['num_pb'].values[0]

    _df['motor'] = motor

    return _df

def apply_parallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = list(tqdm(p.imap(func, [group for _, group in dfGrouped]), total=len(dfGrouped)))

    return pd.concat(ret_list)

if __name__ == '__main__':
    df_compilerates = apply_parallel(df_contractionspeed.groupby(['motor','filename','ATP (uM)', 'pluronic']), separate_replicates)

df_compilerates.to_csv('../../analyzed_data/compiledrates_allmotors_separatereplicates.csv', sep=',')
#df_compilerates.to_csv('../../analyzed_data/compiledrates_pluronic_separatereplicates.csv', sep=',')
# %%
