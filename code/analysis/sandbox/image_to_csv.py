#%%
import numpy as np
import os
from skimage import io
import pandas as pd
import active_matter_pkg as amp

data_root = '../../../data/active_stress/photobleach_data'
directories = np.sort([os.path.join(data_root,dir) for dir in os.listdir(data_root) if 'slide' in dir])
data_dir = str([directory for directory in directories if '210520_slide1_lane3_pos2_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_25ms_DLPBlue_skip1_frame24_photobleach_1' in directory][0])
_, _, mt_trimmed, _, subdirectory = amp.io.tiff_walk(os.path.join(data_dir), parse_channels=True)

df_info = amp.io.parse_filename(data_dir)
df_grat = pd.read_csv('../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
pxl_to_mu = df_grat['micron_per_pixel'].values[0]
# %%
im0 = io.imread(mt_trimmed[0])
x = np.arange(1,np.shape(im0)[1]+1,1) * pxl_to_mu
y = np.arange(1,np.shape(im0)[0]+1,1) * pxl_to_mu
Y,X = np.meshgrid(y,x)

df = pd.DataFrame([])
for n in range(1):
    im = io.imread(mt_trimmed[n])
    _df = pd.DataFrame(np.transpose([np.ravel(X),np.ravel(Y)]),
                        columns=['x','y'])
    _df['u'] = np.ravel(im)
    #_df['t'] = n * df_info['time interval (s)'].values[0]
    df = df.append(_df,ignore_index=True)
# %%
df.to_csv(os.path.join(data_dir,'image_as_csv.txt'),sep=',',index=False)
# %%
df2 = pd.DataFrame([])
for n in range(len(mt_trimmed)):
    im = io.imread(mt_trimmed[n])
    _df = pd.DataFrame([[n*df_info['time interval (s)'].values[0],im]],
                        columns=['time','image'])
    df2 = df2.append(_df,ignore_index=True)
# %%
df2.to_csv(os.path.join(data_dir,'image_as_csv2.csv'),sep=',',index=False)
# %%
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(df['x'],df['y'],df['u'], cmap='viridis')
# %%
plt.imshow(im)
io.imsave(os.path.join(data_dir,'test_mt_trimmed.jpg'),im)
# %%
