#%%
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import integrate, special
from tqdm import tqdm
from numba import njit
from multiprocess import Pool, cpu_count
import active_matter_pkg as amp
amp.viz.plotting_style()

# Define parameters of the system
L = 50 # µm
D = 0.02 # µm^2/s
alpha = 0.002 # 1/s
gamma = np.sqrt(alpha / D) # 1/µm

x_ph = 10 # µm

x = np.linspace(-L, L, 200)
y = np.linspace(-L, L, 200)
t = np.arange(0, 100, 1)
X,Y,T = np.meshgrid(x,y,t)

def integrate_one(k, gamma, L):
      return [k, integrate.quad(lambda x: np.exp(- (gamma * x)**2/2)*(special.hyp1f1(-k**2/2, 1/2, (gamma * x)**2/2)**2), 0, L)[0]]

def integrate_fn(k):
    return integrate_one(k, gamma, L)

def compute_term(k, integral, gamma, alpha, x_ph, X, Y, T):
        hypx = special.hyp1f1(-k**2/2, 1/2, (gamma * X)**2/2)
        hypy = special.hyp1f1(-k**2/2, 1/2, (gamma * Y)**2/2)

        time = np.exp(- alpha * k**2 * T) 
        prefactor = (special.hyp1f1(-k**2/2, 3/2, (gamma * x_ph)**2 / 2)) / integral

        return prefactor * time * hypx, prefactor * time * hypy

@njit(parallel=True)
def compute_singlearggpu(k_arr):
    x_base = 1
    y_base = 1

    for k_h in k_arr:
        k, integral = k_h
        x_int, y_int = compute_term(k, integral, gamma, alpha, x_ph, X, Y, T)
        x_base += x_int
        y_base += y_int

    return x_base, y_base

def compute_singlearg(k_arr):
    k, integral = k_arr
     
    return compute_term(k, integral, gamma, alpha, x_ph, X, Y, T)

def apply_parallel(array, func):
    with Pool(cpu_count() - 1) as p:
        ret_list = list(tqdm(p.imap(func, array), total=len(array)))

    return np.array(ret_list)

#%%
k = np.linspace(0,500,50000)
hyper1f1 = special.hyp1f1(1-k**2/2, 3/2, (gamma * L)**2 / 2)

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
plt.show()
#plt.savefig('../../write_ups/frap/telescoping/hypergeometric_1d_diff_crossing.pdf', bbox_inches='tight',
#            background_color='white')
#%%
k_h = zeros

if __name__ == '__main__':
    integral = apply_parallel(k_h, integrate_fn)

if __name__ == '__main__':
    terms = apply_parallel(integral, compute_singlearg)

x_base, y_base = np.sum(terms, axis=0)

x_base += 1
y_base += 1

c = (x_ph * gamma)**2 * np.exp(- gamma**2 * (X**2 + Y**2) / 2) * (x_base * y_base)

#%%
plt.imshow(c[:,:,0])
# %%
xslice = 101
t_step = np.array([0,1])
blues = cm.Blues_r(np.linspace(0.2,0.6,len(t_step)))
frap = np.zeros(len(x))
frap = np.where((x <= x_ph) & (x >= -x_ph), 1, 0)

fig, ax = plt.subplots(1,2,figsize=(16,8))

ax[0].plot(Y[:,xslice,0],c[:,xslice,0], lw=2, label=r'$t=0$', color='b')
ax[0].plot(x,frap, label='initial condition', color='k')
ax[0].set_title('FRAP conditions', fontsize=24)

for n in t_step:
        ax[1].plot(Y[:,xslice,n],c[:,xslice,n], lw=2, label=r'$t=%.3f$' %t[n],
                color=blues[np.where(t_step==n)[0][0]])
ax[1].set_title('FRAP recovery', fontsize=24)

for a in ax:
        a.set_ylim([-0.1, 1.1])
        a.set_xlim([-L, L])
        a.set_xlabel(r'$x$', fontsize=20)
        h = a.set_ylabel(r'$\frac{c(r,t)}{c_0}$', fontsize=24)
        h.set_rotation(0)
ax[0].legend(loc=1, fontsize=16)
ax[1].legend(loc=1, ncol=1, fontsize=16)
#ax[0].text(-0.1, 1.05, '(A)', fontsize=20)
#ax[1].text(-0.1,1.05, '(B)', fontsize=20)
#plt.savefig('../../figures/advection_diffusion_linearvel1d.pdf',
#        bbox_inches='tight', background_color='white')

# %%
