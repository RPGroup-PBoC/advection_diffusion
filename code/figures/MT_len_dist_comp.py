#%%
import sys
sys.path.insert(0, '../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import active_matter_pkg
active_matter_pkg.viz.plotting_style()
import seaborn as sns

# Pixel conversion
pixel_size = 0.161

df_sh = pd.read_csv('../../analyzed_data/MT_lengths.csv')
df_sh['dataset'] = 'Aug 2020'
df_sh['color'] = 'dodgerblue'
df_sh['style'] = 'o'

df_apr_hj = pd.read_csv('../../analyzed_data/MT_lengths_midapril2021_old_method.csv')
df_apr_hj['dataset'] = 'Apr 2021 (1)'
df_apr_hj['color'] = 'green'
df_apr_hj['style'] = 'x'

df_apr_sh = pd.read_csv('../../analyzed_data/MT_lengths_midapril2021_SH_method.csv')
df_apr_sh['dataset'] = 'Apr 2021 (2)'
df_apr_sh['color'] = 'tomato'
df_apr_sh['style'] = 's'

df_sept22 = pd.read_csv('../../analyzed_data/MT_lengths_sept2022.csv')
df_sept22['dataset'] = 'Sept 2022'
df_sept22['color'] = 'rebeccapurple'
df_sept22['style'] = 'P'

df_j22 = pd.read_csv('../../analyzed_data/MT_lengths_june2022.csv', sep=',')
df_j22['dataset'] = 'Jun 2022'
df_j22['color'] = 'black'
df_j22['style'] = 'd'

df = pd.concat([df_sh, df_apr_hj, df_apr_sh, df_j22, df_sept22])

fig, ax = plt.subplots(1,2,figsize=(16,8))

for set,_df in df.groupby('dataset'):

        d = _df[_df['perimeter']>3]
        ecdf = np.sort(d['perimeter'].values) * pixel_size
        dataset = d['dataset'].values[0]
        y = np.arange(1, len(ecdf)+1, 1) / len(ecdf)
        vals, bins = np.histogram(d['perimeter'].values * pixel_size,
                                  bins = np.linspace(0, 15, 20))
        frac = vals / len(d['perimeter'])
        ax[0].scatter((bins[1:] + bins[:-1]) / 2, frac, 
                      s=40, color=d['color'].values[0], marker=d['style'].values[0])
        ax[0].scatter(np.nan, np.nan, color=d['color'].values[0],
                      s=60, marker=d['style'].values[0],
                      label='%s, median=%.1f µm, n=%i' %(set, np.median(d['perimeter'].values * pixel_size), len(d)))
        ax[1].scatter(ecdf, y, color=d['color'], s=20, marker=d['style'].values[0])
        ax[1].scatter(np.nan, np.nan, color=d['color'].values[0],
                      s=60, marker=d['style'].values[0],
                      label='%s, median=%.1f µm, n=%i' %(set, np.median(d['perimeter'].values * pixel_size), len(d)))


for a in ax:
    a.set_xlabel('length [µm]', fontsize=18)
ax[0].set_ylabel('fraction of MTs', fontsize=18)
ax[1].set_ylabel('ECDF', fontsize=18)
ax[0].legend(loc=1, fontsize=16)
ax[1].legend(loc=4, fontsize=16)

ax[0].set_xlim([0.0, 15])
ax[0].set_xscale('linear')

ax[1].set_xlim([np.min(df['perimeter'])*pixel_size + 0.2, np.max(df['perimeter'])*pixel_size])
ax[1].set_xscale('log')

ax[0].text(-0.4, 0.385, '(A)', fontsize=22, ha='right', va='bottom')
ax[1].text(0.3, 1.048, '(B)', fontsize=22, ha='right', va='bottom')

plt.savefig('../../figures/SIFigX_MT_lengths_SH_paper.pdf', bbox_inches='tight',
            facecolor='white')
