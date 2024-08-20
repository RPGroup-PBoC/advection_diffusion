#%%
import sys
sys.path.insert(0, '../')
import active_matter_pkg as amp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
amp.viz.plotting_style()
# %%

n_ppc_samples = 4000
percent_vals = [0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99]

#%%
# Perform prior predictive checks on linear regression
# Draw parameters out of prior on our contraction rate
slope = np.random.normal(0.001, 0.00025, size=n_ppc_samples)
sigma = np.abs(np.random.normal(0.001, 0.5, size=n_ppc_samples))
b = np.random.normal(0, 0.01, size=n_ppc_samples)
x = np.linspace(0, 100, 100)
# Draw data sets out of the likelihood for each set of prior params
df_standard = pd.DataFrame(columns=['x','simulated y','slope','sigma','b'])
for sl,sig,b_ in zip(slope, sigma, b):
    for xi in x:
        sim_data = np.array(np.random.normal(sl * xi + b_, sig, size=1))
        _df = pd.DataFrame(sim_data, columns={'simulated y'})
        _df['slope'] = sl
        _df['sigma'] = sig
        _df['b'] = b_
        _df['x'] = xi
        df_standard = pd.concat([df_standard,_df], ignore_index=True)

#%%
# quantiles
df_perc = pd.DataFrame(columns={'x', 'quantile', 'quantile low', 'quantile high'})
for g,d in df_standard.groupby('x'):
    for perc in percent_vals:
        quantile_low = np.quantile(d['simulated y'], 0.5 - perc/2)
        quantile_high = np.quantile(d['simulated y'], 0.5 + perc/2)
        _df = pd.DataFrame([g], columns={'x'})
        _df['quantile'] = perc
        _df['quantile low'] = quantile_low
        _df['quantile high'] = quantile_high
        df_perc = pd.concat([df_perc,_df], ignore_index=True)
fig, ax = plt.subplots(1, 1, figsize=(8,8))
for g,d in df_perc.groupby('quantile'):
    ax.fill_between(d['x'], d['quantile low'], d['quantile high'], 
                    color='tomato', alpha=0.4, label='%.2f' %g)
for g,d in df_perc.groupby('x'):
    ax.scatter(g, d['simulated y'].mean(), color='dodgerblue', s=80)
ax.set_xlabel('radius', fontsize=16)
ax.set_ylabel('contraction speed', fontsize=16)
#ax.set_xlim([0, 2.00])
##            bbox_inches='tight', facecolor='w')
# %%
# Perform similar prior predictive checks on sample data
# Draw parameters out of prior on extract sample
psi = np.random.uniform(0, 1/30.0, size=n_ppc_samples)
sigma_v = np.abs(np.random.normal(0, 0.1, size=n_ppc_samples))
beta = np.random.normal(0, 0.1, size=n_ppc_samples)
vol = np.linspace(10, 25, 4)

# Draw data sets out of the likelihood for each set of prior params
df_sample = pd.DataFrame(columns=['volume','simulated abs', 'psi', 'sigma_v', 'beta'])
for ps,sig,beta_ in zip(psi,sigma_v,beta):
    for vi in vol:
        sim_data = np.array(np.random.normal(ps * vi + beta_, sig, size=1))
        _df = pd.DataFrame(sim_data, columns={'simulated abs'})
        _df['psi'] = ps
        _df['sigma_v'] = sig
        _df['beta'] = beta_
        _df['volume'] = vi
        df_sample = df_sample.append(_df, ignore_index=True)

# Look at different quantile levels
df_perc = pd.DataFrame(columns={'volume', 'quantile', 'quantile low', 'quantile high'})
for g,d in df_sample.groupby('volume'):
    for perc in percent_vals:
        quantile_low = np.quantile(d['simulated abs'], 0.5 - perc/2)
        quantile_high = np.quantile(d['simulated abs'], 0.5 + perc/2)
        _df = pd.DataFrame([g], columns={'volume'})
        _df['quantile'] = perc
        _df['quantile low'] = quantile_low
        _df['quantile high'] = quantile_high
        df_perc = df_perc.append(_df, ignore_index=True)
fig, ax = plt.subplots(1,1,figsize=(8,8))
for g,d in df_perc.groupby('quantile'):
    ax.fill_between(d['volume'], d['quantile low'], d['quantile high'],
                    color='tomato', alpha=0.4, label='%.2f' %g)
for g,d in df_sample.groupby('volume'):
    ax.scatter(g, d['simulated abs'].mean(),
            color='dodgerblue', s=80)
ax.set_xlabel('sample volume (µL)', fontsize=16)
ax.set_ylabel('absorbance', fontsize=16)
ax.set_xlim([10.0, 25.0])