#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import scipy.optimize
import pandas as pd
import os
import active_matter_pkg as amp
amp.viz.plotting_style()

data_root1 = '../../../data/active_stress/photobleach_data/210519_slide2_lane2_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1/'
data_root2 = '../../../data/active_stress/photobleach_data/210423_slide1_lane2_pos1/210423_slide1_lane2_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_200ms_DLPYellow_200ms_DLPRed_50ms_DLPBlue_skip1_frame3_photobleach_1/'
data_root3 = '../../../data/active_stress/photobleach_data/210520_slide1_lane2_pos2/210520_slide1_lane2_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_50ms_DLPYellow_50ms_DLPRed_25ms_DLPBlue_skip1_frame80_photobleach_1'

data_roots = [data_root1, data_root2, data_root3]
# %%
df = pd.DataFrame()
for root in data_roots:
    d = pd.read_csv(os.path.join(root,'cross_correlation.csv'))
    df = df.append(d, ignore_index=True)
df = df.sort_values(['photobleach frame','frames after photobleach'])
# %%
def exponential(t, tau, alpha):
    return alpha * np.exp(- t / tau)
#%%
filenames = df['filename'].unique()
colors = ['rebeccapurple', 'dodgerblue', 'tomato']
color_mapping = dict(zip(filenames,colors))

fig, ax = plt.subplots(2,1,figsize=(6,8))
for f,d in df.groupby('photobleach frame'):
    dt = d['intervals (sec)'].values[0]
    t_pb = f * dt

    # renormalize frames
    corr = filters.gaussian(d['correlation sum'].values, sigma=5)
    d['correlation normalized'] = (corr - corr.min()) / (corr.max() - corr.min())

    # Trying 1/e thresholding
    tau_guess = np.argwhere(np.diff(np.sign(d['correlation normalized']-np.exp(-1)))).flatten()[0]
    p_guess = np.array([tau_guess, 1])
    
    p_opt, p_cov = scipy.optimize.curve_fit(exponential, d['frames after photobleach'].values,
                                            d['correlation normalized'].values, p0=p_guess)

    ax[0].plot(dt * d['frames after photobleach'].values, d['correlation normalized'].values,
            color=color_mapping[d['filename'].values[0]], label=f)
    ax[1].scatter(t_pb, dt * p_opt[0], color=color_mapping[d['filename'].values[0]])
ax[0].set_xlabel('time after photobleach [sec]', fontsize=16)
ax[0].set_ylabel('normalized correlation', fontsize=16)
ax[0].legend(loc=1, fontsize=14)

ax[1].set_xlabel('time of photobleach (sec)', fontsize=16)
ax[1].set_ylabel(r'recovery time $\tau$ (sec)', fontsize=16)
ax[1].set_xlim([0,1600])
ax[1].set_ylim([0,750])
plt.savefig('../figures/FigX_correlation_time.pdf', bbox_inches='tight',
            facecolor='white')
# %%
fig, ax = plt.subplots(1,1,figsize=(6,6))
for f,d in df.groupby('photobleach frame'):
    dt = d['intervals (sec)'].values[0]
    t_pb = f * dt

    # renormalize frames
    corr = filters.gaussian(d['correlation sum'].values, sigma=5)
    d['correlation normalized'] = (corr - corr.min()) / (corr.max() - corr.min())

    # Trying 1/e thresholding
    tau_guess = np.argwhere(np.diff(np.sign(d['correlation normalized']-np.exp(-1)))).flatten()[0]
    p_guess = np.array([tau_guess, 1])
    
    p_opt, p_cov = scipy.optimize.curve_fit(exponential, d['frames after photobleach'].values,
                                            d['correlation normalized'].values, p0=p_guess)

    ax.scatter(t_pb, dt * p_opt[0], color=color_mapping[d['filename'].values[0]])

ax.set_xlabel('time of photobleach (sec)', fontsize=16)
ax.set_ylabel(r'recovery time $\tau$ (sec)', fontsize=16)
ax.set_xlim([0,1600])
ax.set_ylim([0,750])
plt.savefig('../figures/FigX_correlation_tim_v2.pdf', bbox_inches='tight',
            facecolor='white')
# %%
