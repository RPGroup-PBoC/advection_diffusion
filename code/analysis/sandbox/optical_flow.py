#%%
# Try looking at correlation length
import os
import active_matter_pkg as amp
from skimage import io, filters, feature, morphology, measure, registration, color
from scipy import optimize, spatial
import numpy as np
from scipy import ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
amp.viz.plotting_style()
%matplotlib inline

root = '../../../data/active_stress/photobleach_data'
data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])

data_root = str([directory for directory in data_directory if '210520_slide1_lane2_pos2' in directory][0])
#data_root = str([directory for directory in data_directory if '210519_slide2_lane2_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1' in directory][0])
#data_root = str([directory for directory in data_directory if '2018' in directory][0])

mt_imgfiles, mot_imgfiles, mt_trimmed, mot_trimmed, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

if len(subdirectory)>0:
    if any('filename_order.csv' in filename for filename in os.listdir(data_root)):
        df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
        data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    if any('image_crop.csv' in filename for filename in os.listdir(data_root)):
        df_crop = pd.read_csv(os.path.join(data_root,'image_crop.csv'), sep=',')
df_info = amp.io.parse_filename(data_root)
df_graticule = pd.read_csv('../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
um_per_pxl = df_graticule['micron_per_pixel'].values[0]

df_t2c = pd.read_csv('../analyzed_data/time_to_contraction.csv', sep=',')
#n_contract = df_t2c[df_t2c['filename']==data_root]['frame_num'].values[0]
#mt_trunc = [file for file in mt_imgfiles if data_root in file]
#num_pb = 53
if int(os.path.split(data_root)[1][:6]) < 210520:
    num_pb = df_info['photobleach frame number'].values[0]
else:
    num_pb = df_info['photobleach frame number'].values[0]-1
im_first = io.imread(mt_imgfiles[0])
im_before_pb = io.imread(mt_imgfiles[num_pb-1])
# At this time, the first two images taken after photobleaching are 
# taken with little interval in between. Skipping the first of the two
im_pb1 = io.imread(mt_imgfiles[num_pb+1])
im_pb2 = io.imread(mt_imgfiles[num_pb+2])

dict_color = dict({-1:'green',0:'rebeccapurple',1:'tomato',2:'dodgerblue'})

# bounding box restricting primarily to activation region
x_camcent = 610
y_camcent = 975
if num_pb < 60:
    winsize_cam = 450
else:
    winsize_cam = 300
#winsize_cam = 525

window = np.s_[int(x_camcent-winsize_cam):int(x_camcent+winsize_cam),int(y_camcent-winsize_cam):int(y_camcent+winsize_cam)]
fig, ax = plt.subplots(2,2,figsize=(16,16))
ax[0,0].imshow(im_before_pb[window], cmap='flag')
ax[0,1].imshow(io.imread(mt_imgfiles[num_pb])[window], cmap='flag')
ax[1,0].imshow(im_pb1[window])
ax[1,1].imshow(im_pb2[window])

for a in ax:
    for _a in a:
        _a.set_xticklabels([])
        _a.set_yticklabels([])

plt.show()

if 'optical_flow_analyzed' not in os.listdir(data_root):
    os.mkdir(os.path.join(data_root,'optical_flow_analyzed'))

if 'optical_flow_csv' not in os.listdir(data_root):
    os.mkdir(os.path.join(data_root,'optical_flow_csv'))

# Should fit to sinusoidal function of alpha * np.sin(theta - theta_0)
def sinusoid(theta, alpha, theta_0, offset):
    return alpha * np.sin(theta - theta_0) + offset

def find_angle_offset(angle, v_theta, tolerance=0.05):
    alpha_guess = np.abs((v_theta.max() - v_theta.min())/2)

    theta_peak = angle[np.where(v_theta==v_theta.max())[0]][0]
    theta_trough = angle[np.where(v_theta==v_theta.min())[0]][0]

    if theta_peak > theta_trough:
        theta_guess = (theta_peak - theta_trough)/2
    else:
        theta_guess = (theta_peak - theta_trough + np.pi)/2
    offset_guess = 0

    p_guess = [alpha_guess, theta_guess, offset_guess]

    p_opt, _ = optimize.curve_fit(sinusoid, angle, v_theta, p0=p_guess)

    if p_opt[0] < 0 and theta_guess > 0:
        p_opt[0] *= -1
        p_opt[1] -= np.pi
    elif p_opt[0] < 0 and theta_guess < 0:
        p_opt[0] *= -1
        p_opt[1] += np.pi

    if p_opt[0] < tolerance:
        return np.nan
    elif p_opt[0] > tolerance:
        return p_opt[0:2]

def go_with_flow(v_x, v_y, center, step_size=10):
    x_center = center[0]
    x1 = int(np.ceil(x_center))
    x0 = int(np.floor(x_center))
    dx = x_center - x0
    
    y_center = center[1]
    y1 = int(np.ceil(y_center))
    y0 = int(np.floor(y_center))
    dy = y_center - y0

    x_center2 = x_center + ((1-dx)*(1-dy)*v_x[y0,x0] + dx*(1-dy)*v_x[y0,x1] + (1-dx)*dy*v_x[y1,x0] + dx*dy*v_x[y1,x1]) * step_size
    y_center2 = y_center + ((1-dx)*(1-dy)*v_y[y0,x0] + dx*(1-dy)*v_y[y0,x1] + (1-dx)*dy*v_y[y1,x0] + dx*dy*v_y[y1,x1]) * step_size
    return x_center2, y_center2, np.sqrt((x_center2-x_center)**2 + (y_center2-y_center)**2)

def find_sink(v_x, v_y, center, step_size=1, tol=0.1, n_iterations=20):
    n_iter = 0
    dr = step_size
    while (dr > tol) or (n_iter<n_iterations):
        x_sink, y_sink, dr = go_with_flow(v_x, v_y, center, step_size=step_size)
        center = np.array([x_sink, y_sink])
        n_iter += 1
        if dr > 10*tol:
            n_iter = 0
    return x_sink, y_sink, dr, n_iter
#%%
step=30
n=0
im1 = io.imread(mt_trimmed[n])
im1_norm = amp.image_processing.normalize(im1)
im2 = io.imread(mt_trimmed[n+1])
im2_norm = amp.image_processing.normalize(im2)
offset=1914
v_y, v_x = registration.optical_flow_tvl1(im1_norm,im2_norm,
                                        tightness=0.1,attachment=50)
v_y -= np.average(v_y)
v_x -= np.average(v_x)

im_gauss = filters.gaussian(im1,sigma=20) * (2**16-1)
im_subt = im1 - im_gauss
thresh = filters.threshold_mean(im_subt)
im_binary = (im_subt > thresh)
#square = morphology.square(3)
#im_closed = morphology.closing(im_binary, selem=square)

im_label = measure.label(im_binary)
# Remove smalls and objects near borders
im_border = amp.image_processing.border_clear(im_label, edge=5)
im_small = amp.image_processing.remove_small(im_border, area_thresh=400)
im_bw = (im_small > 0)

im_relabel, n_labels = measure.label(im_bw, return_num=True)

im_filled = np.zeros(np.shape(im1))
square = morphology.square(3)
for m in range(1,n_labels+1):
    im_nclosed = morphology.closing(im_relabel==m, selem=square)
    
    edges = feature.canny(im_nclosed)
    edges_closed = morphology.closing(edges, selem=square)
    filled_cell = ndi.binary_fill_holes(edges_closed)
    im_filled += filled_cell * m
if num_pb<60:
    regprops_center = measure.regionprops_table((im_filled>0).astype(int), intensity_image=im1,
                                        properties=['weighted_centroid'])
    df_center = pd.DataFrame(regprops_center)

    x_cent = df_center['weighted_centroid-1'].values[0]
    y_cent = df_center['weighted_centroid-0'].values[0]

    x_cent, y_cent, dr, n_iter = find_sink(v_x, v_y, np.array([x_cent,y_cent]), step_size=10, tol=0.01)
else:
    if len(mot_imgfiles)>0:
        im_motor = io.imread(mot_trimmed[n]) - offset
        im_mot_gauss = filters.gaussian(im_motor, sigma=5)
        popt, _ = amp.image_processing.gaussian_fit(im_mot_gauss,im_mot_gauss.max())
        x_cent, y_cent = popt[0], popt[1]
    else:
        im_blurred = filters.gaussian(im1,sigma=20)
        popt, _ = amp.image_processing.gaussian_fit(im_blurred, im_blurred.max())
        x_cent, y_cent = popt[0], popt[1]

x = np.arange(0,np.shape(im1)[0],1)
y = np.arange(0,np.shape(im1)[1],1)
X,Y = np.meshgrid(x,y)

radii = np.sqrt((X-x_cent)**2 + (Y-y_cent)**2)
v_r = ((X-x_cent)*v_x + (Y - y_cent)*v_y)/(radii)
mag = np.sqrt(v_x**2 + v_y**2)

fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(im1)
ax[0].quiver(X[::step,::step], Y[::step,::step],
        v_x[::step,::step],v_y[::step,::step],
        edgecolor='dodgerblue', units='dots', angles='xy',
        scale_units='xy', lw=3, alpha=0.8)
ax[0].scatter(x_cent,y_cent, color='tomato', s=20)
ax[1].scatter(np.ravel(radii[::5,::5]),np.ravel(v_r[::5,::5]), 
            color='dodgerblue', alpha=0.01, rasterized=True)
plt.show()
#%%
# Plotting radial and angular components of velocity as a function
# of radius. Change to lighter colors for later frames
steps = 10

df = pd.DataFrame([])
plt.clf()
for imset in data_directory:
    _, _, mt_trimmed, mot_trimmed, subdirectory = amp.io.tiff_walk(imset, parse_channels=True)
    if len(subdirectory)>0:
        if any('filename_order.csv' in str(filename) for filename in os.listdir(imset)):
            df_csv = pd.read_csv(os.path.join(imset, os.path.split(imset)[-1]+ '_filename_order.csv'), sep=',')
            imset_root = df_csv[df_csv['order']==df_csv['order'].max()]['filename'].values[0]
        if any('image_crop.csv' in filename for filename in os.listdir(imset_root)):
            df_crop = pd.read_csv(os.path.join(data_root,'image_crop.csv'), sep=',')
    else:
        imset_root = imset.copy()
    if len(mt_trimmed)==0:
        continue
    df_info = amp.io.parse_filename(imset_root)
    dt = df_info['time interval (s)'].values[0]
    num_pb = df_info['photobleach frame number'].values[0]
    
    im1 = io.imread(mt_trimmed[0])
    im2 = io.imread(mt_trimmed[1])

    im1_norm = amp.image_processing.normalize(im1)
    im2_norm = amp.image_processing.normalize(im2)

    v_y, v_x = registration.optical_flow_tvl1(im1,im2,tightness=0.1,
                                            attachment=50)
    v_y -= np.average(v_y)
    v_x -= np.average(v_x)

    im_gauss = filters.gaussian(im1,sigma=20) * (2**16-1)
    im_subt = im1 - im_gauss
    thresh = filters.threshold_mean(im_subt)
    im_binary = (im_subt > thresh)
    #square = morphology.square(3)
    #im_closed = morphology.closing(im_binary, selem=square)

    im_label = measure.label(im_binary)
    # Remove smalls and objects near borders
    im_border = amp.image_processing.border_clear(im_label, edge=5)
    im_small = amp.image_processing.remove_small(im_border, area_thresh=400)
    im_bw = (im_small > 0)

    im_relabel, n_labels = measure.label(im_bw, return_num=True)

    im_filled = np.zeros(np.shape(im1))
    im_mask = np.zeros(np.shape(im1))
    square = morphology.square(3)
    for m in range(1,n_labels+1):
        im_nclosed = morphology.closing(im_relabel==m, selem=square)
        
        edges = feature.canny(im_nclosed)
        edges_closed = morphology.closing(edges, selem=square)
        filled_cell = ndi.binary_fill_holes(edges_closed)
        im_filled += filled_cell * m
        im_mask += filled_cell

    idx_overlap = np.argwhere(im_mask>1)
    for idx in idx_overlap:
        im_filled[idx[0],idx[1]] = 0
    
    if num_pb<60:
        regprops_center = measure.regionprops_table((im_filled>0).astype(int), intensity_image=im1,
                                            properties=['weighted_centroid'])
        df_center = pd.DataFrame(regprops_center)
        x_cent = df_center['weighted_centroid-1'].values[0]
        y_cent = df_center['weighted_centroid-0'].values[0]
    
        x_sink, y_sink, dr, n_iter = find_sink(v_x, v_y, np.array([x_cent,y_cent]), step_size=10, tol=0.01)
    else:
        if len(mot_trimmed)>0:
            im_motor = io.imread(mot_trimmed[0]) - offset
            im_mot_gauss = filters.gaussian(im_motor, sigma=5)
            popt, _ = amp.image_processing.gaussian_fit(im_mot_gauss,im_mot_gauss.max())
            x_cent, y_cent = popt[0], popt[1]
            x_sink = np.nan
            y_sink = np.nan
    regprops = measure.regionprops_table(im_filled.astype(int), intensity_image=im1,
                                        properties=['area','centroid','label', 'intensity_image',
                                                    'weighted_centroid','mean_intensity'])
    _df = pd.DataFrame(regprops)

    if n>0:
        dist = spatial.distance.cdist(_df[['centroid-0','centroid-1']],df[df['time']==n-1][['centroid-0','centroid-1']])
        id_idx = [np.argwhere(dist[i,:]==dist[i,:].min())[0][0] for i in range(len(dist[:,0]))]
        id = df[df['time']==n-1]['cellID'].values[id_idx]
        for _id in range(len(id)):
            if dist[_id,id_idx[_id]] > 20:
                id[_id] = id_max
                id_max += 1
            if _df[_df['label']==_id+1]['area'].values[0] > 4000:
                id[_id] = 99

        _df['cellID'] = id

    _df['time'] = n
    _df['total_intensity'] = [np.sum(entry) for entry in _df['intensity_image'].values[:]]
    if num_pb < 80:
        _df['network_xcent'] = x_sink
        _df['network_ycent'] = y_sink
    else:
        _df['network_xcent'] = x_cent
        _df['network_ycent'] = y_cent

    if n==0:
        _df['cellID'] = _df['label']
        id_max = _df['label'].max() + 1

    _df = _df.drop(columns=['intensity_image'])

    Dxx_avg = np.zeros(len(_df))
    Dxy_avg = np.zeros(len(_df))
    Dyy_avg = np.zeros(len(_df))
    Dxx_std = np.zeros(len(_df))
    Dxy_std = np.zeros(len(_df))
    Dyy_std = np.zeros(len(_df))

    for idx in range(1,int(im_filled.max())):
        mask_m = (im_filled==idx)

        pxl = np.where(mask_m)

        dxy, dxx = np.gradient(filters.gaussian(v_x,sigma=3))/dt
        dyy, dyx = np.gradient(filters.gaussian(v_y,sigma=3))/dt

        Dxx_avg[idx] = np.average(dxx[pxl[0],pxl[1]])
        Dxx_std[idx] = np.std(dxx[pxl[0],pxl[1]])
        Dxy_avg[idx] = np.average((dxy[pxl[0],pxl[1]] + dyx[pxl[0],pxl[1]])/2)
        Dxy_std[idx] = np.std((dxy[pxl[0],pxl[1]] + dyx[pxl[0],pxl[1]])/2)
        Dyy_avg[idx] = np.average(dyy[pxl[0],pxl[1]])
        Dyy_std[idx] = np.std(dyy[pxl[0],pxl[1]])
    _df['Dxx'] = Dxx_avg
    _df['Dxx std'] = Dxx_std
    _df['Dxy'] = Dxy_avg
    _df['Dxy std'] = Dxy_std
    _df['Dyy'] = Dyy_avg
    _df['Dyy std'] = Dyy_std
    _df['filename'] = imset
    _df['photobleach time'] = num_pb

    df = df.append(_df, ignore_index=True)

#%%
lower_plot_bound = df[['Dxx','Dxy','Dyy']].min().min()
upper_plot_bound = df[['Dxx','Dxy','Dyy']].max().max()
df['radius'] = np.sqrt((df['weighted_centroid-1']-df['network_xcent'])**2 + (df['weighted_centroid-0']-df['network_ycent'])**2) * um_per_pxl
for t,d in df.groupby('photobleach time'):
    _,_,mt_trimmed,_,subdirectory = amp.io.tiff_walk(d['filename'].values[0], parse_channels=True)
    fig, ax = plt.subplots(2,2,figsize=(16,16))
    ax[0,0].imshow(io.imread(mt_trimmed[0]))
    ax[0,0].scatter(d['network_xcent'],d['network_ycent'], color='tomato', s=20)

    ax[0,1].scatter(d['radius'],d['Dxx'], color='dodgerblue')
    ax[0,1].errorbar(d['radius'],d['Dxx'],yerr=d['Dxx std'],
                    color='dodgerblue', linestyle='None')
    ax[1,0].scatter(d['radius'],d['Dxy'])
    ax[1,0].errorbar(d['radius'],d['Dxy'],yerr=d['Dxy std'],
                    color='dodgerblue', linestyle='None')
    ax[1,1].scatter(d['radius'],d['Dyy'])
    ax[1,1].errorbar(d['radius'],d['Dyy'],yerr=d['Dyy std'],
                    color='dodgerblue', linestyle='None')
    #ax[0,1].set_title(r'$D_{xx}$', fontsize=20)
    #ax[1,0].set_title(r'$D_{xy}$', fontsize=20)
    #ax[1,1].set_title(r'$D_{yy}$', fontsize=20)
    for _,_d in d.groupby('label'):
        ax[0,0].text(_d['centroid-1'].values[0],_d['centroid-0'].values[0],
                    '%i' %_d['cellID'], color='white', ha='center',va='center')
        ax[0,1].text(_d['radius'],_d['Dxx'],'  %i' %_d['cellID'], ha='left', va='bottom')
        ax[1,0].text(_d['radius'],_d['Dxy'],'  %i' %_d['cellID'], ha='left', va='bottom')
        ax[1,1].text(_d['radius'],_d['Dyy'],'  %i' %_d['cellID'], ha='left', va='bottom')
    ax[0,0].set_title('photobleach frame = %i' %t)
    ax[0,1].set_xlabel('radius [μm]', fontsize=20)
    ax[1,0].set_xlabel('radius [μm]', fontsize=20)
    ax[1,1].set_xlabel('radius [μm]', fontsize=20)
    ax[0,1].set_ylabel(r'$D_{xx}$ [1/s]', fontsize=20)
    ax[1,0].set_ylabel(r'$D_{xy}$ [1/s]', fontsize=20)
    ax[1,1].set_ylabel(r'$D_{yy}$ [1/s]', fontsize=20)
    #ax[0,1].set_ylim(lower_plot_bound,upper_plot_bound)
    #ax[1,0].set_ylim(lower_plot_bound,upper_plot_bound)
    #ax[1,1].set_ylim(lower_plot_bound,upper_plot_bound)
    ax[0,0].axes.get_xaxis().set_visible(False)
    ax[0,0].axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig('../figures/coarse_grained_strain_rate_numpb%i.pdf' %t,
                bbox_inches='tight', background_color='tight')

# %%
# Look at difference between two different frames.
im1 = io.imread(mt_trimmed[2])
im1 = color.rgb2gray(amp.image_processing.normalize(im1))
im2 = io.imread(mt_trimmed[3])
im2 = color.rgb2gray(amp.image_processing.normalize(im2))
nr, nc = np.shape(im1)
seq_im = np.zeros((nr,nc,3))
seq_im[...,0]=im2
seq_im[...,1]=im1
seq_im[...,2]=im1
plt.figure(figsize=(8,8))
plt.imshow(seq_im)
# %%
