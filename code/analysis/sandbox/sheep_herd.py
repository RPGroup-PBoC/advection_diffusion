#%%
import sys
sys.path.insert(0,'../')
import os
import active_matter_pkg as amp
import imageio_ffmpeg
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from skimage import io, filters, feature, morphology, segmentation
amp.viz.plotting_style()
# %%
root = '~/Downloads/sheep_herd_2.mp4'
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
imset = imageio_ffmpeg.read_frames(root, pix_fmt='rgb24')

for im in imset:
  im
# %%


# %%
