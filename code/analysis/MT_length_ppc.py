#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import active_matter_pkg.viz
import bebi103
import bokeh
active_matter_pkg.viz.plotting_style()
# %%
n_ppc_samples = 1000

# Draw parameters out of hte prior
log_lambda = np.random.uniform(np.log(0.01),np.log(100), size=n_ppc_samples)
lam = np.exp(log_lambda)

# Draw data sets out of the likelihood for lambda
ell = np.array([np.random.exponential(scale=l, size=900) for l in lam])
# %%
fig, ax = plt.subplots(1,1,figsize=(8,8))
y = np.linspace(0, 900, 900) / 900
y[-1] = 1

for e in ell[9::10]:
    e = np.sort(e)
    ax.plot(e, y, color='tomato', lw=2, alpha=0.1)
    ax.set_xscale('log')
# %%
