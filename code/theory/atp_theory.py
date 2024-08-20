#%%
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import active_matter_pkg as amp
amp.viz.plotting_style()

x = np.linspace(0.001, 1000, 100000)
speed = x / (1 + x)

fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot(x, speed, color='dodgerblue')
ax.set_xscale('log')
ax.set_xlabel(r'$\frac{[A]}{K_d}$', fontsize=16)
ax.set_ylabel(r'$\frac{v_\mathrm{motor}}{d \, \gamma_\mathrm{max}}$        ', fontsize=16, rotation=0)
plt.savefig('../../SIFigX_atp_motorspeed.pdf', bbox_inches='tight',
            facecolor='white')
# %%
