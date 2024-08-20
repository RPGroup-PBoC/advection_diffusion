#%%
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import active_matter_pkg as amp
amp.viz.plotting_style()

datanames = ['../../analyzed_data/2d_advdiff_gridlines_0deg_extracoarse.txt',
        '../../analyzed_data/2d_advdiff_gridlines_0deg_coarse.txt',
        '../../analyzed_data/2d_advdiff_gridlines_0deg_normal.txt',
        '../../analyzed_data/2d_advdiff_gridlines_0deg_fine.txt',
        '../../analyzed_data/2d_advdiff_gridlines_0deg_extrafine.txt',
        '../../analyzed_data/2d_advdiff_gridlines_0deg_extremefine.txt']
mesh = ['extremely coarse', 'coarse', 'normal',
        'fine', 'extra fine', 'extremely fine']
df = pd.DataFrame()
for n in range(len(datanames)):
        _df = pd.read_csv(datanames[n],skiprows=7,sep=';')
        col = [col_ for col_ in _df.columns if 'cln1x' in col_]
        _df = _df.rename(columns={col[0]:'r'})
        _df['mesh'] = mesh[n]
        _df['number'] = n
        df = df.append(_df, ignore_index=True)

# %%
colors = ['dodgerblue', 'tomato', 'rebeccapurple',
        'green','orange','brown']
color_dict = dict(zip(mesh,colors))
hw = 1.25
cent = np.array([0,5,10])
r = np.linspace(0,10,100)
c0 = np.zeros(100)
c0[r<1.25] = 1
c0[(r>3.75) & (r<6.25)] = 1
c0[(r>8.75)] = 1

fig, ax = plt.subplots(2,3,figsize=(18,12))
for m,d in df.groupby('mesh'):
        col = [col_ for col_ in d.columns if 't=0' in col_]
        _col = int(d['number'].values[0]%3)
        row = int(np.floor(d['number'].values[0]/3))
        ax[row,_col].plot(d['r'], d[col[0]] * 10**7, color='dodgerblue', 
                        lw=2, label='FEM')
        ax[row,_col].set_title(m, fontsize=20)
for a in ax.flatten():
        a.plot(r,c0,color='black',ls='--', zorder=-1, label='true initial condition')
        a.set_xlim(0,10)
        a.set_ylim(-0.05,1.45)
for a in ax[-1,:]:
        a.set_xlabel('radius [μm]', fontsize=24)
for a in ax[:,0]:
        a.set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)
ax[0,0].legend(loc='upper left', fontsize=14)
for a in ax.flatten():
        a.tick_params(axis='both',labelsize=20)

ax[0,0].text(0.0,1.46,'(A)', ha='left',va='bottom', fontsize=20)
ax[0,1].text(0.0,1.46,'(B)', ha='left',va='bottom', fontsize=20)
ax[0,2].text(0.0,1.46,'(C)', ha='left',va='bottom', fontsize=20)
ax[1,0].text(0.0,1.46,'(D)', ha='left',va='bottom', fontsize=20)
ax[1,1].text(0.0,1.46,'(E)', ha='left',va='bottom', fontsize=20)
ax[1,2].text(0.0,1.46,'(F)', ha='left',va='bottom', fontsize=20)

fig.tight_layout()
#plt.savefig('../../figures/advection_diffusion_comsol_gibbs.pdf',
#        bbox_inches='tight', background_color='white')
# %%
