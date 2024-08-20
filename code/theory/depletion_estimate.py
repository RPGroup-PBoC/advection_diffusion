#%%
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import active_matter_pkg as amp
amp.viz.plotting_style()

x = np.linspace(0, 1, 1000)
force = 15 / 8 * np.sqrt(x)

_, ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(x, force, color='dodgerblue')
ax.text(-0.45, 1.15, 'depletion force', fontsize=20)
ax.set_ylabel(r'$\frac{F}{c k_B T L \left[2 (R + r) r \right]^{1/2}}$                     ', rotation=0)
ax.set_xlabel(r'$\left[ \frac{\epsilon}{2r} \right]$')

plt.savefig('../../figures/SIFigX_depletion_force.pdf', bbox_inches='tight',
            facecolor='white')
# %%
