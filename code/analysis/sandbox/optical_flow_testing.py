#%%
# Trial functions for optical flow
import os
import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage import filters, morphology, measure, registration, draw, transform
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import active_matter_pkg as amp
amp.viz.plotting_style()

def gaussian_sym(x,y,x_cent,y_cent,sigma):
    gauss = np.exp(-((x-x_cent)**2+(y-y_cent)**2)/(sigma**2))
    return gauss/gauss.max()

def donut(x,y,x_cent,y_cent,sigma,parabolic_width,offset=0):
    donut = (((x-x_cent)**2 + (y-y_cent)**2) / parabolic_width + offset) * np.exp(-((x-x_cent)**2+(y-y_cent)**2)/(sigma**2))
    return donut / donut.max()

def compute_strains(v_x,v_y,sigma=None):
    if sigma!=None:
        v_x = filters.gaussian(v_x,sigma=sigma)
        v_y = filters.gaussian(v_y,sigma=sigma)

    Dxx, dxy = np.gradient(v_x)
    dyx, Dyy = np.gradient(v_y)

    Dxy = (dxy + dyx) / 2
    return Dxx, Dxy, Dyy

step = 20
#%%
# Generate disk of same intensity throughout
n_frames = 10
radii = np.linspace(200,400,n_frames)

x = np.arange(0,900,1)
y = np.arange(0,900,1)
Y,X = np.meshgrid(y,x)

radial = np.sqrt((Y - 450)**2 + (X - 450)**2)

circles = np.zeros((900,900,n_frames))

for frame in range(n_frames):
    r,c = draw.disk((450,450),radii[n_frames-1-frame])
    circles[r,c,frame] = 1
#%%
v_rmax = np.zeros(n_frames-1)
v_err = np.zeros((2,n_frames-1))
r_err = np.zeros((2,n_frames-1))
n_positions = 100
dr = radial.max()/n_positions

for n in range(n_frames-1):
    v_y, v_x = registration.optical_flow_tvl1(circles[:,:,n],circles[:,:,n+1], tightness=0.03, attachment=5)
    mag = np.sqrt(v_y**2 + v_x**2)
    r_intervals, v_avg, v_std = amp.stats.avg_profile(mag, radial, x_min=0, x_max=radial.max(), 
                                                    n_positions=n_positions, dx=dr)
    
    #v_err will compute the median for values lower than the max radius and for values higher than the max radius
    # separately
    v_err[0,n] = np.median(mag[radial < radii[n_frames-1-n]])
    r_err[0,n] = radial[radial < radii[n_frames-1-n]].flatten()[np.where(mag[radial < radii[n_frames-1-n]]==v_err[0,n])[0]][0]
    v_err[1,n] = np.median(mag[radial > radii[n_frames-1-n]])
    r_err[1,n] = radial[radial > radii[n_frames-1-n]].flatten()[np.where(mag[radial > radii[n_frames-1-n]]==v_err[1,n])[0]][0]

    v_rmax[n] = np.mean(mag[(radial > radii[n_frames-n-1] - dr/2) & (radial < radii[n_frames-n-1] + dr/2)])

    if n==6:
        fig, ax = plt.subplots(1,2,figsize=(16,8))
        ax[0].imshow(circles[:,:,n])
        ax[0].quiver(Y[::step,::step],X[::step,::step], v_x[::step,::step], v_y[::step,::step],
                    color='dodgerblue',  units='dots', angles='xy', scale_units='xy')
        ax[1].scatter(radial[::step,::step],mag[::step,::step],color='dodgerblue', alpha=0.05)
        ax[1].scatter(np.nan,np.nan, color='dodgerblue', label='optical flow')
        ax[1].axvline(radii[n_frames-1-n],-10,10,color='tomato',ls='--', label='outer radius')
        ax[1].axhline(radii[n_frames-1-n]-radii[n_frames-2-n], -10, 10, color='green', ls='--', label='expected velocity')
        ax[1].set_xlabel('radius', fontsize=20)
        ax[1].set_ylabel(r'$| \vec{v} |$', fontsize=20, rotation=0)
        ax[1].legend(loc=4, fontsize=16)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        #ax[1].set_ylim(-0.1,6.1)
        plt.savefig('../figures/SIFigX_circle_contraction_frame%i.pdf' %n, bbox_inches='tight')
#%%
v_rmax = np.zeros(n_frames-1)
v_err = np.zeros((2,n_frames-1))
r_err = np.zeros((2,n_frames-1))
n_positions = 100
dr = radial.max()/n_positions

for n in range(10):
    v_y, v_x = registration.optical_flow_tvl1(circles[:,:,n],circles[:,:,n+1], tightness=0.01, attachment=5)
    mag = np.sqrt(v_y**2 + v_x**2)
    r_intervals, v_avg, v_std = amp.stats.avg_profile(mag, radial, x_min=0, x_max=radial.max(), 
                                                    n_positions=n_positions, dx=dr)
    
    #v_err will compute the median for values lower than the max radius and for values higher than the max radius
    # separately
    v_err[0,n] = np.median(mag[radial < radii[n_frames-1-n]])
    r_err[0,n] = radial[radial < radii[n_frames-1-n]].flatten()[np.where(mag[radial < radii[n_frames-1-n]]==v_err[0,n])[0]][0]
    v_err[1,n] = np.median(mag[radial > radii[n_frames-1-n]])
    r_err[1,n] = radial[radial > radii[n_frames-1-n]].flatten()[np.where(mag[radial > radii[n_frames-1-n]]==v_err[1,n])[0]][0]

    v_rmax[n] = np.mean(mag[(radial > radii[n_frames-n-1] - dr/2) & (radial < radii[n_frames-n-1] + dr/2)])

    if n<10:
        fig, ax = plt.subplots(1,2,figsize=(16,8))
        ax[0].imshow(circles[:,:,n])
        ax[0].quiver(Y[::step,::step],X[::step,::step], v_x[::step,::step], v_y[::step,::step],
                    color='dodgerblue',  units='dots', angles='xy', scale_units='xy')
        ax[1].scatter(radial[::step,::step],mag[::step,::step],color='dodgerblue', alpha=0.05)
        ax[1].scatter(np.nan,np.nan, color='dodgerblue', label='optical flow')
        ax[1].axvline(radii[n_frames-1-n],-10,10,color='tomato',ls='--', label='outer radius')
        ax[1].axhline(radii[n_frames-1-n]-radii[n_frames-2-n], -10, 10, color='green', ls='--', label='expected velocity')
        ax[1].set_xlabel('radius', fontsize=20)
        ax[1].set_ylabel(r'$| \vec{v} |$', fontsize=20, rotation=0)
        ax[1].legend(loc=4, fontsize=16)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        #ax[1].set_ylim(-0.1,6.1)
        #plt.savefig('../figures/SIFigX_circle_contraction_final_frame%i.pdf' %n, bbox_inches='tight')
#%%
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.scatter(radii[:-1], radii[1:]-radii[:-1], color='tomato', label='expected values')
ax.scatter(radii[1:][::-1], v_rmax, color='dodgerblue', label='optical flow')

ax.scatter(r_err, v_err, color='rebeccapurple', label='extent of flow smoothness')

ax.set_xlabel('radius', fontsize=18)
ax.set_ylabel('velocity', fontsize=18)
ax.set_xlim(150,450)
ax.set_ylim(0,25)
ax.legend(loc=4)
# %%
# Generate dataset for a contracting circle with increasing but uniform intensity with each successive frame
n_frames = 10
radii = np.linspace(100,400,n_frames)

x = np.arange(0,900,1)
y = np.arange(0,900,1)
Y,X = np.meshgrid(y,x)

radial = np.sqrt((Y - 450)**2 + (X - 450)**2)

circles = np.zeros((900,900,n_frames))

for frame in range(n_frames):
    r,c = draw.disk((450,450),radii[n_frames-1-frame])
    if frame==0:
        norm = len(r)
    circles[r,c,frame] = norm / len(r)

for n in range(n_frames-1):
    v_y, v_x = registration.optical_flow_tvl1(circles[:,:,n],circles[:,:,n+1], tightness=0.01, attachment=5)
    mag = np.sqrt(v_y**2 + v_x**2)
    r_intervals, v_avg, v_std = amp.stats.avg_profile(mag, radial, x_min=0, x_max=radial.max(), 
                                                    n_positions=n_positions, dx=dr)
    if n==6:
        fig, ax = plt.subplots(1,2,figsize=(16,8))
        ax[0].imshow(circles[:,:,n])
        ax[0].quiver(Y[::step,::step],X[::step,::step], v_x[::step,::step], v_y[::step,::step],
                    color='dodgerblue',  units='dots', angles='xy', scale_units='xy')
        ax[1].scatter(radial[::step,::step],mag[::step,::step],
                    color='dodgerblue', alpha=0.05)
        ax[1].scatter(np.nan,np.nan,color='dodgerblue', label='optical flow')
        ax[1].axvline(radii[n_frames-1-n],-10,10,color='tomato',ls='--', label='outer radius')
        ax[1].axhline(radii[n_frames-1-n]-radii[n_frames-2-n], -10, 10, color='green', ls='--', label='expected velocity')
        ax[1].errorbar(r_intervals, v_avg, yerr=v_std, color='rebeccapurple', label='average velocity')
        ax[1].set_xlabel('radius', fontsize=20)
        ax[1].set_ylabel(r'$| \vec{v} |$', fontsize=20, rotation=0)
        ax[1].legend(loc=1, fontsize=16)
        ax[1].set_ylim(np.min(v_avg-v_std)-10,np.max(v_avg+v_std)+10)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        #ax[1].set_ylim(-0.1,6.1)
        
        plt.savefig('../figures/SIFigX_circle_accumulating_frame%i.pdf' %n, bbox_inches='tight')
        plt.show()

# %%
Dxx, Dxy, Dyy = compute_strains(v_x, v_y)

selem = np.ones((3,3))
fig, ax = plt.subplots(1,3,figsize=(18,6))
ax[0].imshow(filters.median(Dxx, selem=selem), cmap='viridis')
ax[1].imshow(filters.median(Dxy, selem=selem), cmap='viridis')
ax[2].imshow(filters.median(Dyy, selem=selem), cmap='viridis')
# %%
gaussians = np.zeros((900,900,n_frames))

x_cent = 450
y_cent = 450
sigma = np.linspace(100, 200, n_frames)

for n in range(n_frames):
    noise = np.random.random((900,900))*0.1
    gaussians[:,:,n] = gaussian_sym(X,Y,x_cent,y_cent,sigma[n_frames-n-1]) + noise
    gaussians[:,:,n] /= gaussians[:,:,n].max()
# %%
for n in range(6,7):
    v_y, v_x = registration.optical_flow_tvl1(gaussians[:,:,n],gaussians[:,:,n+1], tightness=0.01,attachment=20)
    mag = np.sqrt(v_y**2 + v_x**2)

    if n==6:
        r = np.linspace(0,2*sigma[n_frames-1-n],200)
        fit = r * (1 - sigma[n_frames-n-2]/sigma[n_frames-n-1])

        circ = patches.Circle((450,450), radius=2*sigma[n_frames-n-1], fill=False, ec='tomato', ls='--', lw=2)

        fig, ax = plt.subplots(1,2,figsize=(16,8))
        ax[0].imshow(gaussians[:,:,n])
        ax[0].quiver(Y[::step,::step],X[::step,::step], v_x[::step,::step], v_y[::step,::step],
                    color='dodgerblue',  units='dots', angles='xy', scale_units='xy')
        ax[1].scatter(radial[::step,::step],mag[::step,::step], color='dodgerblue', alpha=0.05)
        ax[1].plot(r,fit,color='green', lw=2,ls='--', label='expected fit')
        ax[1].axvline(2*sigma[n_frames-1-n], color='tomato', lw=2, ls='--', label=r'$2 \sigma$')
        ax[1].legend(loc=1, fontsize=20)
        ax[1].set_xlabel('radius [px]', fontsize=20)
        ax[1].set_ylabel(r'$| \vec{v} |$', fontsize=20, rotation=0)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)

        ax[0].add_patch(circ)
        #if n==6:
        plt.savefig('../figures/SIFigX_gaussian_optical_flow_frame%i.pdf' %n,
                    bbox_inches='tight')
        plt.show()

#%%
Dxx, Dxy, Dyy = compute_strains(v_x, v_y)

fig, ax = plt.subplots(2,3,figsize=(18,12))
ax[0,0].imshow(Dxx, cmap='viridis')
ax[0,1].imshow(Dxy, cmap='viridis')
ax[0,2].imshow(Dyy, cmap='viridis')

ax[1,0].scatter(radial[radial < radii[n]][::step], Dxx[radial < radii[n]][::step], cmap='dodgerblue', alpha=0.01)
ax[1,1].scatter(radial[radial < radii[n]], Dxy[radial < radii[n]], cmap='dodgerblue', alpha=0.01)
ax[1,2].scatter(radial[radial < radii[n]], Dyy[radial < radii[n]], cmap='dodgerblue', alpha=0.01)
# %%
n_frames = 10
donuts = np.zeros((900,900,n_frames))

x_cent = 450
y_cent = 450
sigma = np.linspace(50, 200, n_frames)

for n in range(n_frames):
    noise = np.random.random((900,900))*0.1
    donuts[:,:,n] = donut(X,Y,x_cent,y_cent,sigma[n_frames-n-1],1, offset=0) + noise
    donuts[:,:,n] /= donuts[:,:,n].max()
    if n < 10:
        plt.figure(figsize=(8,8))
        plt.imshow(donuts[:,:,n])
# %%
for n in range(7):
    v_y, v_x = registration.optical_flow_tvl1(donuts[:,:,n],donuts[:,:,n+1],
                                            tightness=0.01,attachment=30)
    mag = np.sqrt(v_x**2 + v_y**2)

    if n==2 or n==6:

        r = np.linspace(0, 2*sigma[n_frames-n-1], 1000)
        delta = r * (1 - sigma[n_frames-n-2]/sigma[n_frames-n-1])
        outline = patches.Circle((450,450),radius=2*sigma[n_frames-n-1], color='tomato', lw=2, ls='--', fill=False)
        fig, ax = plt.subplots(2,1,figsize=(8,16))
        ax[0].imshow(donuts[:,:,n])
        ax[0].quiver(Y[::step,::step],X[::step,::step], v_x[::step,::step], v_y[::step,::step],
                    color='dodgerblue',  units='dots', angles='xy', scale_units='xy')
        ax[1].scatter(radial[::step,::step],mag[::step,::step], color='dodgerblue', alpha=0.05)
        ax[1].scatter(np.nan,np.nan,color='dodgerblue', label='optical flow')
        ax[1].plot(r, delta, color='green', ls='--', lw=2, label='expected output')
        ax[1].axvline(2*sigma[n_frames-1-n], -10, 100, color='tomato', lw=2, ls='--', label=r'$r=2\sigma$')
        ax[1].legend(loc=1, fontsize=20)
        ax[1].set_xlabel('radius [px]', fontsize=20)
        ax[1].set_ylabel(r'$| \vec{v} |$   ', fontsize=20, rotation=0)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        ax[0].add_patch(outline)
        plt.savefig('../figures/SIFigX_donut_optical_flow_frame%i.pdf' %n,
                    bbox_inches='tight')
        plt.show()

# %%
Dxx, Dxy, Dyy = compute_strains(v_x, v_y)

fig, ax = plt.subplots(2,3,figsize=(18,12))
ax[0,0].imshow(Dxx, cmap='viridis')
ax[0,1].imshow(Dxy, cmap='viridis')
ax[0,2].imshow(Dyy, cmap='viridis')

ax[1,0].scatter(radial[radial < radii[n]], Dxx[radial < radii[n]], cmap='dodgerblue')
ax[1,1].scatter(radial[radial < radii[n]], Dxy[radial < radii[n]], cmap='dodgerblue')
ax[1,2].scatter(radial[radial < radii[n]], Dyy[radial < radii[n]], cmap='dodgerblue')
# %%
