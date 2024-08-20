"""Image processing for pixel size on camera, 20x"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
import bokeh

import active_matter_pkg.viz
active_matter_pkg.viz.plotting_style()
bokeh.io.output_notebook()

# Locate data root directory
data_root = "../../../data/active_stress/"

# Specify graticule location
graticule_path = 'microscope_setup_sample_images/200123_graticule/Pos0/img_000000000_Default_000.tif'

graticule_20x = io.imread(os.path.join(data_root,graticule_path))

# %%
io.imshow(graticule_20x[150:450,:])

# %%
_ = plt.plot(graticule_20x[150:450,:].T)

# %%
graticule_slice = graticule_20x[200,:]

# Smooth data to ensure one local minimum
grat_smooth = ndimage.filters.gaussian_filter1d(graticule_slice, 2)

# Set thresholding below 30000 for pixel intensity
intensity_threshold = 30000
# %%
# Apply local minimum method
lcl_min_bool = np.r_[True, grat_smooth[1:] < grat_smooth[:-1]] & np.r_[grat_smooth[:-1] < grat_smooth[1:], True] & np.r_[grat_smooth < intensity_threshold]
lcl_min_vals = grat_smooth[lcl_min_bool]

# Find indices of trues
indices = [ind for ind, lcl_min in enumerate(lcl_min_bool) if lcl_min]
indices = np.asarray(indices)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(grat_smooth, color='dodgerblue')
ax.scatter(indices, lcl_min_vals, color='tomato', lw=0)

# %%
# Calculate difference
pixel_distance = indices[1:] - indices[:-1]
mean_pxl_dist = np.mean(pixel_distance)

# %%
pxl_per_micron_20x = mean_pxl_dist / 10.0
micron_per_pxl_20x = 10.0 / mean_pxl_dist

df = pd.DataFrame(np.array([[pxl_per_micron_20x, micron_per_pxl_20x]]),
                    columns=['pxl_per_micron', 'micron_per_pixel'])
df['objective'] = 20
df.to_csv('./objective_pxl_micron_scale.csv')
# %%
