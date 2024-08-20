#%%
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import integrate, special
from tqdm import tqdm
import active_matter_pkg as amp
amp.viz.plotting_style()
# %%
k = np.linspace(0,300,1000000)
hyper1f1 = special.hyp1f1(1-k**2/2, 3/2, 1/2)

zeros = []

for i in range(len(k)-1):
    if hyper1f1[i]*hyper1f1[i+1]<0:
        zeros.append((k[i] + k[i+1])/2)
zeros = np.array(zeros)

plt.plot(k, hyper1f1)
plt.scatter(zeros, zeros*0, s=100, color='tomato')
plt.axhline(0, 0, 100, color='k', linestyle='--')
plt.xlim([0,k.max()])
plt.xlabel(r'${\tilde k}$', fontsize=20)
plt.ylabel(r'$_1F_1( 1 - \frac{{\tilde k}^2}{2}; \frac{3}{2} ; \frac{1}{2})$',
        fontsize=20)
#plt.savefig('../../write_ups/frap/telescoping/hypergeometric_1d_diff_crossing.pdf', bbox_inches='tight',
#            background_color='white')
# %%
x_ph = 0.5
k_h = zeros
integral = [integrate.quad(lambda x: np.exp(-x**2/2)*(special.hyp1f1(-k**2/2,1/2,x**2/2)**2), 0, 1) for k in k_h]

x = np.linspace(0, 1, 100)
t = np.linspace(0, 0.5, 100)
X,T = np.meshgrid(x,t)
c = np.exp(- X**2 / 2) * (1 - special.erf(x_ph/np.sqrt(2)) / special.erf(1/np.sqrt(2)))

for n in tqdm(range(len(k_h))):
        hyp = np.exp(- X**2 /2 ) * special.hyp1f1(-k_h[n]**2/2,1/2,X**2/2)
        time = np.exp(- k_h[n]**2 * T) 
        prefactor = (x_ph * np.exp(- x_ph**2 / 2) * special.hyp1f1(1 - k_h[n]**2/2,3/2,x_ph**2/2))/integral[n][0]
        c -= prefactor * time * hyp
# %%
t_step = np.array([0,4,8,16,20,99])
blues = cm.Blues_r(np.linspace(0.2,0.6,len(t_step)))
frap = np.zeros(len(x))
frap = np.where(x > x_ph, np.exp(-x**2/2), 0)

fig, ax = plt.subplots(1,2,figsize=(16,8))
ax[0].plot(x, np.exp(- x**2 / 2), color='r', ls='--',
        label=r'$t < 0$', zorder=10)
ax[0].plot(X[0,:],c[0,:], lw=2, label=r'$t=0$', color='b')
ax[0].plot(x,frap, label='initial condition', color='k')
ax[0].set_title('FRAP conditions', fontsize=24)

ax[1].plot(x, np.exp(- x**2 / 2), color='r', ls='--',
        label=r'$t < 0$', zorder=10)
for n in t_step:
        ax[1].plot(X[n,:],c[n,:], lw=2, label=r'$t=%.3f$' %t[n],
                color=blues[np.where(t_step==n)[0][0]])
ax[1].set_title('FRAP recovery', fontsize=24)

for a in ax:
        a.set_ylim([-0.01, 1.01])
        a.set_xlim([0, 1])
        a.set_xlabel(r'$x$', fontsize=20)
        h = a.set_ylabel(r'$\frac{c(r,t)}{c_0}$', fontsize=24)
        h.set_rotation(0)
ax[0].legend(loc=4, fontsize=16)
ax[1].legend(loc=4, ncol=2, fontsize=16)
ax[0].text(-0.1, 1.05, '(A)', fontsize=20)
ax[1].text(-0.1,1.05, '(B)', fontsize=20)
#plt.savefig('../../figures/advection_diffusion_linearvel1d.pdf',
#        bbox_inches='tight', facecolor='white')
# %%
