#%%
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
import os
import active_matter_pkg as amp
amp.viz.plotting_style()

root = '../../analyzed_data/Schief2003PNAS_AllFig.csv'

df = pd.read_csv(root, sep=',')
#filename = [f for f in os.listdir(os.path.join(root, directory_list[0]))]

# %%
df_noadp = df[(df['ADP (uM)'] == 0) & (df['P (uM)']==0) & (df['fig']=='2')]
plt.scatter(df_noadp['ATP (uM)'], df_noadp['Speed (nm/s)'],
         color='k', s=10)
plt.xscale('log')
# %%
def hill_fn(x, alpha, Kd):
    return alpha * (x / Kd) / (1 + x / Kd)

alpha0 = df_noadp['Speed (nm/s)'].max()
Kd0 = 50

p0 = [alpha0, Kd0]

popt_k401, _ = curve_fit(hill_fn, df_noadp['ATP (uM)'],
                    df_noadp['Speed (nm/s)'], p0 = p0)

x_theor = np.logspace(np.log10(df_noadp['ATP (uM)'].min()), np.log10(df_noadp['ATP (uM)'].max()), 100)
y_theor = hill_fn(x_theor, *popt_k401)

plt.plot(x_theor, y_theor, color='tomato')
plt.scatter(df_noadp['ATP (uM)'], df_noadp['Speed (nm/s)'],
            color='dodgerblue', s=12)
plt.xlabel('ATP concentration [µM]')
plt.ylabel('motor speed [nm/s]')
plt.xscale('log')
plt.savefig('../../figures/SIFigX_atp_k401speed.pdf',
            bbox_inches='tight', facecolor='white')
# %%
df_glides = pd.read_csv('../../analyzed_data/20231117_aggregateGliding', sep=',')
df_cleaned = df_glides[(df_glides['ADP_Conc_uM']==0) & (df_glides['P_Conc_uM']==0) & (df_glides['Motor_Conc_uM']==0.0010)]
df_ncd = pd.DataFrame()

for info, d in df_cleaned.groupby(['ATP_Conc_uM', 'Motor_Conc_uM', 'MT_Conc_uM', 'Motor Type']):
    atp, motor, mt, species = info

    _df_ncd = pd.DataFrame([[d['speed (nm/s)'].mean() - 5.52]],
                           columns=['mean speed (nm/s)'])
    
    _df_ncd['ATP (uM)'] = atp
    _df_ncd['Motor (uM)'] = motor
    _df_ncd['MT (uM)'] = mt
    _df_ncd['Motor Type'] = species

    df_ncd = pd.concat([df_ncd, _df_ncd], ignore_index=True)
#df_ncd['mean speed (nm/s)'] -= df_ncd['mean speed (nm/s)'].min()
#%%
plt.scatter(df_ncd['ATP (uM)'], df_ncd['mean speed (nm/s)'])
plt.xscale('log')
# %%
def ncd_fn(x, kdet, kD):
    return (-3.8 * kdet**2 * kD * x + 5.2 * (kdet * (kD * x)**2)) / (kdet**2 + kdet * kD * x + (kD * x)**2)

p0 = [40, 0.8]

popt_ncd, _ = curve_fit(ncd_fn, df_ncd['ATP (uM)'], df_ncd['mean speed (nm/s)'],
                    p0 = p0, bounds=((0, 0), (np.inf, np.inf)))

x_ncd = np.logspace(np.log10(df_ncd['ATP (uM)'].min()),
                    np.log10(df_ncd['ATP (uM)'].max()),
                    100)
y_ncd = ncd_fn(x_ncd, *popt_ncd)

plt.scatter(df_ncd['ATP (uM)'], df_ncd['mean speed (nm/s)'],
            color='dodgerblue', s=12)
plt.plot(x_ncd, y_ncd, color='tomato')
plt.xlabel('ATP concentration [µM]')
plt.ylabel('motor speed [nm/s]')
plt.xscale('log')
plt.savefig('../../figures/SIFigX_atp_ncdspeed.pdf',
            bbox_inches='tight', facecolor='white')
# %%
df_parameters = pd.DataFrame([popt_ncd],
                             columns=('k_detach', 'k_ATP'))
df_parameters['motor'] = 'ncd'

_df_parameters = pd.DataFrame([popt_k401],
                              columns=('alpha', 'k_ATP'))
_df_parameters['motor'] = 'k401'

df_parameters = pd.concat([df_parameters, _df_parameters],
                          ignore_index=False)
df_parameters.to_csv('../../analyzed_data/atp_speed_parameters.csv',
                     sep=',')
# %%
