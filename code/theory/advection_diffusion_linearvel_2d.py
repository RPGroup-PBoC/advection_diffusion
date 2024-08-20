#%%
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from scipy import integrate, special
import active_matter_pkg as amp
amp.viz.plotting_style()

datanames = ['../../analyzed_data/2d_advdiff_uniformic_t0_20_40_80_160_990.txt',
        '../../analyzed_data/2d_advdiff_uniform2ic_t0_20_40_80_160_990.txt',
        '../../analyzed_data/2d_advdiff_gaussianic_t0_20_40_80_160_990.txt']
ic = ['uniform', 'uniform2', 'gaussian']
df = pd.DataFrame()
for n in range(len(datanames)):
        _df = pd.read_csv(datanames[n],skiprows=7,sep=',')
        col = [col_ for col_ in _df.columns if 'cln1x' in col_]
        _df = _df.rename(columns={col[0]:'r'})
        _df['initial'] = ic[n]
        df = pd.concat([df,_df], ignore_index=True)

# %%
Rmax = 10
v_max = 0.1
D = 0.1
l_ = np.sqrt(D * Rmax / v_max)
k_max = 100
k = np.linspace(-0.0001,k_max,1000000)
hyper1f1_2d = k**2 * special.hyp1f1(1 -(k**2)/2, 2, Rmax**2/(2*l_**2))

zeros = []

for i in range(len(k)-1):
    if hyper1f1_2d[i]*hyper1f1_2d[i+1]<0:
        zeros.append((k[i] + k[i+1])/2)
zeros = np.array(zeros)

k_h = np.unique(zeros) / l_

plt.plot(k, hyper1f1_2d)
plt.scatter(0, 0, s=100, color='tomato')
plt.scatter(zeros, zeros*0, s=100, color='tomato')
plt.axhline(0, 0, 100, color='k', linestyle='--')
plt.xlim([0,k_max])
plt.xlabel(r'${\tilde k}$', fontsize=20)
plt.ylabel(r'$_1F_1( 1 - \frac{{\tilde k}^2}{2}; 2 ; 10)$', fontsize=20)
#plt.savefig('../../figures/SIFigX_hypergeometric_2d_diff_crossing.pdf', bbox_inches='tight',
#            background_color='white')

# %%
# Calculation for uniform initial condition
R0_u = Rmax/2
R0_g = Rmax/4
integral = [integrate.quad(lambda r: r * np.exp(-r**2/(2 * l_**2))*(special.hyp1f1(-l_**2 * k**2/2,1,r**2/(2 * l_**2))**2), 0, Rmax) for k in k_h]

r = np.linspace(0, Rmax, int(100*Rmax))
t = np.linspace(0, 10000, 1000)
R,T = np.meshgrid(r,t)
c_uni1 = 1/2 * (Rmax**2 / l_**2) * np.exp(- R**2 / (2 * l_**2)) / (1 - np.exp(-Rmax**2/(2 * l_**2)))
c_uni2 = 1/2 * ((Rmax**2 / l_**2) - (R0_u**2 / l_**2)) * np.exp(- R**2 / (2 * l_**2)) / (1 - np.exp(-Rmax**2/(2 * l_**2)))
c_gauss = (np.exp(-R0_g**2/(2 * l_**2)) - np.exp(- Rmax**2 / (2*l_**2))) * np.exp(- R**2 / (2 * l_**2)) / (1 - np.exp(-Rmax**2/(2 * l_**2)))
for n in range(len(k_h)):
        hyp = np.exp(- R**2 / (2 * l_**2)) * special.hyp1f1(-l_**2 * k_h[n]**2/2,1,R**2/(2 * l_**2))
        time = np.exp(- D * k_h[n]**2 * T) 
        prefactor_uni1 = Rmax**2 * (special.hyp1f1(-l_**2 * k_h[n]**2/2,2, Rmax**2/(2* l_**2)))/(2 * integral[n][0])
        c_uni1 += prefactor_uni1 * time * hyp

        prefactor_uni2 = (Rmax**2 * (special.hyp1f1(-l_**2 * k_h[n]**2/2,2, Rmax**2/(2* l_**2))) - R0_u**2 * (special.hyp1f1(-l_**2 * k_h[n]**2/2,2, R0_u**2/(2* l_**2))))/(2 * integral[n][0])
        c_uni2 += prefactor_uni2 * time * hyp

        if n < 25:
                prefactor_gauss = ( - R0_g**2 * (special.hyp1f1(1 + l_**2 * k_h[n]**2/2, 2, -R0_g**2/(2 * l_**2))))/(2 * integral[n][0])
                c_gauss += prefactor_gauss * time * hyp

t_step = np.array([0,2,4,8,16,99])
purples = cm.Purples_r(np.linspace(0.0,0.5,len(t_step)))
blues = cm.Blues_r(np.linspace(0.0,0.5,len(t_step)))
reds = cm.Reds_r(np.linspace(0.0,0.5,len(t_step)))
#%%
fig, ax = plt.subplots(1,3,figsize=(18,6))

ax[0].plot(r, np.linspace(1,1,len(r)), color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)
for n in t_step:
        ax[0].plot(r,c_uni1[n,:], lw=2, label=r'$t=%i$ s' %t[n],
                color=purples[np.where(t_step==n)[0][0]])
for n in t_step:
        d = df[df['initial']=='uniform']
        col = [col_ for col_ in d.columns if 't=%i' %int(10*n) in col_]
        ax[0].scatter(d['r'], d[col[0]]*10**7, label=r'$t=%i$ s (FE)' %t[n],
                edgecolor=purples[np.where(t_step==n)[0][0]], 
                facecolor='None',s=50, marker='o')
ax[0].set_title(r'(A) $c(r,0)=c_0$', fontsize=22)

uni2 = np.zeros(len(r))
uni2 = np.where(r > R0_u, 1, 0)
ax[1].plot(r, uni2, color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)
for n in t_step:
        ax[1].plot(r,c_uni2[n,:], lw=2, label=r'$t=%i$ s' %t[n],
                color=blues[np.where(t_step==n)[0][0]])
for n in t_step:
        d = df[df['initial']=='uniform2']
        col = [col_ for col_ in d.columns if 't=%i' %int(10*n) in col_]
        ax[1].scatter(d['r'], d[col[0]]*10**7, label=r'$t=%i$ s (FE)' %t[n],
                edgecolor=blues[np.where(t_step==n)[0][0]], 
                facecolor='None',s=50, marker='o')
ax[1].set_title(r'(B) $c(r>R_0,0) = c_0$', fontsize=22)

gauss = np.zeros(len(r))
gauss = np.where(r > R0_g, np.exp(-r**2/(2*l_**2)), 0)
ax[2].plot(r, gauss, color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)

for n in t_step:
        ax[2].plot(r,c_gauss[n,:], lw=2, label=r'$t=%i$ s' %t[n],
                color=reds[np.where(t_step==n)[0][0]])

for n in t_step:
        d = df[df['initial']=='gaussian']
        col = [col_ for col_ in d.columns if 't=%i' %int(10*n) in col_]
        ax[2].scatter(d['r'], d[col[0]]*10**7, label=r'$t=%i$ s (FE)' %t[n],
                edgecolor=reds[np.where(t_step==n)[0][0]], 
                facecolor='None',s=50, marker='o')
ax[2].plot(r, np.exp(-r**2 / (2*l_**2)), color='tomato', ls='--',
        label=r'exp$(-\frac{r^2}{2 \lambda^2})$')
ax[2].set_title(r'(C) $c(r>R_0,0) = c_0 \, \mathrm{exp}(-r^2/2\lambda^2)$', fontsize=22)


for a in ax.flatten():
        a.set_xlim([0, Rmax])
        a.legend(loc=1, ncol=2, fontsize=12)
        a.set_xlabel('radius [μm]', fontsize=20)
ax[0].set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)

ax[0].set_ylim([-0.01, c_uni1.max()*1.05])
ax[1].set_ylim([-0.01, c_uni1.max()+0.1])
ax[2].set_ylim([-0.01, 1.02])

ax[0].set_yticks([0.0,1.0,2.0,3.0,4.0,5.0])
ax[0].set_yticklabels([0.0,1.0,2.0,3.0,4.0,5.0])

ax[1].set_yticks([0.0,1.0,2.0,3.0,4.0,5.0])
ax[1].set_yticklabels([0.0,1.0,2.0,3.0,4.0,5.0])

#ax[0].text(0.0,int(c_uni1.max())+0.45, '(A)', fontsize=24, horizontalalignment='left')
#ax[1].text(0.0,int(c_uni1.max())+0.3, '(B)', fontsize=24, horizontalalignment='left')
#ax[2].text(0.0,1.05, '(C)', fontsize=24, horizontalalignment='left')
fig.tight_layout()
plt.savefig('../../figures/advection_diffusion_linearvel2d.pdf',
        bbox_inches='tight', facecolor='white')
# %%
# Discussion about Gibbs phenomenon
c_uni1 = 1/2 * (Rmax**2 / l_**2) * np.exp(- r**2 / (2 * l_**2)) / (1 - np.exp(-Rmax**2/(2 * l_**2)))
c_uni2 = 1/2 * ((Rmax**2 / l_**2) - (R0_u**2 / l_**2)) * np.exp(- r**2 / (2 * l_**2)) / (1 - np.exp(-Rmax**2/(2 * l_**2)))
c_gauss = (np.exp(-R0_g**2/(2 * l_**2)) - np.exp(- Rmax**2 / (2*l_**2))) * np.exp(- r**2 / (2 * l_**2)) / (1 - np.exp(-Rmax**2/(2 * l_**2)))

num_eigen = np.array([0, 4, 24, 99])
colors = ['rebeccapurple','dodgerblue','tomato','green']
color_dict = dict(zip(num_eigen,colors))

fig, ax = plt.subplots(1,3,figsize=(18,6))
ax[0].plot(r, np.linspace(1,1,len(r)), color='k', ls='--', lw=2, label=r'$c(r,0)$', zorder=10)
ax[1].plot(r, uni2, color='k', ls='--', lw=2, label=r'$c(r,0)$', zorder=10)
ax[2].plot(r, gauss, color='k', ls='--', lw=2, label=r'$c(r,0)$', zorder=10)

for n in range(len(k_h)):
        hyp = np.exp(- r**2 / (2 * l_**2)) * special.hyp1f1(-l_**2 * k_h[n]**2/2,1,r**2/(2 * l_**2))
        prefactor_uni1 = Rmax**2 * (special.hyp1f1(-l_**2 * k_h[n]**2/2,2, Rmax**2/(2* l_**2)))/(2 * integral[n][0])
        c_uni1 += prefactor_uni1 * hyp

        prefactor_uni2 = (Rmax**2 * (special.hyp1f1(-l_**2 * k_h[n]**2/2,2, Rmax**2/(2* l_**2))) - R0_u**2 * (special.hyp1f1(-l_**2 * k_h[n]**2/2,2, R0_u**2/(2* l_**2))))/(2 * integral[n][0])
        c_uni2 += prefactor_uni2 * hyp

        prefactor_gauss = (Rmax**2 * (special.hyp1f1(1 + l_**2 * k_h[n]**2/2,2, -Rmax**2/(2* l_**2))) - R0_g**2 * (special.hyp1f1(1 + l_**2 * k_h[n]**2/2,2, -R0_g**2/(2* l_**2))))/(2 * integral[n][0])
        c_gauss += prefactor_gauss * hyp

        if n in num_eigen:
                if n==0:
                        ax[0].plot(r,c_uni1,color=color_dict[n], lw=3, label='%i eigenvalue' %(n+1))
                        ax[1].plot(r,c_uni2,color=color_dict[n], lw=3, label='%i eigenvalue' %(n+1))
                        ax[2].plot(r,c_gauss,color=color_dict[n], lw=3, label='%i eigenvalue' %(n+1))
                elif n>0:
                        ax[0].plot(r,c_uni1,color=color_dict[n], lw=3, label='%i eigenvalues' %(n+1))
                        ax[1].plot(r,c_uni2,color=color_dict[n], lw=3, label='%i eigenvalues' %(n+1))
                        if n == num_eigen[-1]:
                                continue
                        else:
                                ax[2].plot(r,c_gauss,color=color_dict[n], lw=3, label='%i eigenvalues' %(n+1))
for a in ax:
        a.set_xlabel('radius [μm]', fontsize=20)
        if a==ax[2]:
                a.legend(loc=1, fontsize=16)
        else:
                a.legend(loc=4, fontsize=16)
        a.set_xlim(0,Rmax)
ax[0].set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)

ax[2].set_ylim(-0.3,1.0)
ax[0].set_title(r'$c(r,0)=c_0$', fontsize=22)
ax[1].set_title(r'$c(r>R_0,0) = c_0$', fontsize=22)
ax[2].set_title(r'$c(r>R_0,0) = c_0 \, \mathrm{exp}(-r^2/2\lambda^2)$', fontsize=22)

ax[0].text(0.0,2.8, '(A)', fontsize=24, horizontalalignment='left')
ax[1].text(0.0,2.7, '(B)', fontsize=24, horizontalalignment='left')
ax[2].text(0.0,1.03, '(C)', fontsize=24, horizontalalignment='left')
fig.tight_layout()
plt.savefig('../../figures/SIFigX_gibbs_phenomenon.pdf',
        bbox_inches='tight', facecolor='white')

# %%
def hypergeo(a,b,x, num_terms):
        n = np.arange(0, num_terms, 1)
        factorial = special.factorial(n)
        pocha = special.poch(a,n)
        pochb = special.poch(b,n)
        return np.sum(pocha / pochb * x**n / factorial)

def fn_bc(k, l, R, num_terms):
        fn = np.zeros(len(k))

        for n in range(len(fn)):
                fn[n] = (l**2 * k[n]**2) / 2 * hypergeo(1 - (l**2 * k[n]**2) / 2, 2, R**2 / (2 * l**2), num_terms)

        return fn

k = np.linspace(0, 5, 10000)
plt.plot(k,fn_bc(k, l_, Rmax, 1000000))
# %%
t_step = np.array([0,2,4,8,16,99])
purples = cm.Purples_r(np.linspace(0.0,0.5,len(t_step)))
blues = cm.Blues_r(np.linspace(0.0,0.5,len(t_step)))
reds = cm.Reds_r(np.linspace(0.0,0.5,len(t_step)))

fig, ax = plt.subplots(1,1,figsize=(8,4))

ax.plot(r, np.linspace(1,1,len(r)), color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)
for n in t_step:
        d = df[df['initial']=='uniform']
        ax.plot(r,c_uni1[n,:], lw=2, label=r'$t=%i$ s' %t[n],
                color=purples[np.where(t_step==n)[0][0]])
#for n in t_step:
#        col = [col_ for col_ in d.columns if 't=%i' %int(10*n) in col_]
#        ax.scatter(d['r'], d[col[0]]*10**7, label=r'$t=%i$ s (FE)' %t[n],
#                edgecolor=purples[np.where(t_step==n)[0][0]], 
#                facecolor='None',s=50, marker='o')
ax.set_title(r'$c(r,0)=c_0$', fontsize=22)
ax.set_xlabel('radius [μm]', fontsize=20)
ax.set_ylim([-0.01, c_uni1.max()*1.05])
ax.set_yticks([0.0,1.0,2.0,3.0,4.0,5.0])
ax.set_yticklabels([0.0,1.0,2.0,3.0,4.0,5.0])
ax.set_xlim([0, Rmax])
ax.legend(loc=1, ncol=2, fontsize=12)
ax.set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)

plt.savefig('../../figures/uniform_concentration_noFEM.pdf',bbox_inches='tight',
        facecolor='white')

#%%
fig,ax = plt.subplots(1,1,figsize=(8,4))
uni2 = np.zeros(len(r))
uni2 = np.where(r > R0_u, 1, 0)
ax.plot(r, uni2, color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)
for n in t_step:
        d = df[df['initial']=='uniform2']
        ax.plot(r,c_uni2[n,:], lw=2, label=r'$t=%i$ s' %t[n],
                color=blues[np.where(t_step==n)[0][0]])
for n in t_step:
        col = [col_ for col_ in d.columns if 't=%i' %int(10*n) in col_]
        ax.scatter(d['r'], d[col[0]]*10**7, label=r'$t=%i$ s (FE)' %t[n],
                edgecolor=blues[np.where(t_step==n)[0][0]], 
                facecolor='None',s=50, marker='o')
ax.set_title(r'$c(r>R_0,0) = c_0$', fontsize=22)
ax.set_ylim([-0.01, c_uni1.max()+0.1])
ax.set_yticks([0.0,1.0,2.0,3.0,4.0,5.0])
ax.set_yticklabels([0.0,1.0,2.0,3.0,4.0,5.0])
ax.set_xlabel('radius [μm]', fontsize=20)
ax.set_xlim([0, Rmax])
ax.legend(loc=1, ncol=2, fontsize=12)
ax.set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)
plt.savefig('./figures/frap_concentration.pdf',bbox_inches='tight',
        facecolor='white')
#%%
gauss = np.zeros(len(r))
gauss = np.where(r > R0_g, np.exp(-r**2/(2*l_**2)), 0)
fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(r, gauss, color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)

for n in t_step:
        d = df[df['initial']=='gaussian']
        ax.plot(r,c_gauss[n,:], lw=2, label=r'$t=%i$ s' %t[n],
                color=reds[np.where(t_step==n)[0][0]])
for n in t_step:
        col = [col_ for col_ in d.columns if 't=%i' %int(10*n) in col_]
        ax.scatter(d['r'], d[col[0]]*10**7, label=r'$t=%i$ s (FE)' %t[n],
                edgecolor=reds[np.where(t_step==n)[0][0]], 
                facecolor='None',s=50, marker='o')
ax.plot(r, np.exp(-r**2 / (2*l_**2)), color='tomato', ls='--',
        label=r'exp$(-\frac{r^2}{2 \lambda^2})$')
ax.set_title(r'$c(r>R_0,0) = c_0 \, \mathrm{exp}(-r^2/2\lambda^2)$', fontsize=22)

ax.set_xlim([0, Rmax])
ax.legend(loc=1, ncol=2, fontsize=12)
ax.set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)
ax.set_xlabel('radius [μm]', fontsize=20)

ax.set_ylim([-0.01, 1.02])

fig.tight_layout()
plt.savefig('../../figures/frap_gaussian_concentration.pdf',
        bbox_inches='tight', background_color='white')
# %%
t_step = np.array([0,2,4,8,16,99])
purples = cm.Purples_r(np.linspace(0.0,0.5,len(t_step)))
blues = cm.Blues_r(np.linspace(0.0,0.5,len(t_step)))
reds = cm.Reds_r(np.linspace(0.0,0.5,len(t_step)))

fig, ax = plt.subplots(1,1,figsize=(8,4))

ax.plot(r, np.linspace(1,1,len(r)), color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)
ax.set_title(r'$c(r,0)=c_0$', fontsize=22)
ax.set_xlabel('radius [μm]', fontsize=20)
ax.set_ylim([-0.01, c_uni1.max()*1.05])
ax.set_yticks([0.0,1.0,2.0,3.0,4.0,5.0])
ax.set_yticklabels([0.0,1.0,2.0,3.0,4.0,5.0])
ax.set_xlim([0, Rmax])
ax.legend(loc=1, ncol=2, fontsize=12)
ax.set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)

plt.savefig('../../figures/uniform_before_recovery.pdf',bbox_inches='tight',
        facecolor='white')
# %%
fig,ax = plt.subplots(1,1,figsize=(8,4))
uni2 = np.zeros(len(r))
uni2 = np.where(r > R0_u, 1, 0)
ax.plot(r, uni2, color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)
ax.set_title(r'$c(r>R_0,0) = c_0$', fontsize=22)
ax.set_ylim([-0.01, c_uni1.max()+0.1])
ax.set_yticks([0.0,1.0,2.0,3.0,4.0,5.0])
ax.set_yticklabels([0.0,1.0,2.0,3.0,4.0,5.0])
ax.set_xlabel('radius [μm]', fontsize=20)
ax.set_xlim([0, Rmax])
ax.legend(loc=1, ncol=2, fontsize=12)
ax.set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)
plt.savefig('./figures/frap_concentration_before_recovery.pdf',bbox_inches='tight',
        facecolor='white')
#%%
gauss = np.zeros(len(r))
gauss = np.where(r > R0_g, np.exp(-r**2/(2*l_**2)), 0)
fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(r, gauss, color='k', ls='--',
        label=r'$c(r,0)$', zorder=10)

ax.plot(r, np.exp(-r**2 / (2*l_**2)), color='tomato', ls='--',
        label=r'exp$(-\frac{r^2}{2 \lambda^2})$')
ax.set_title(r'$c(r>R_0,0) = c_0 \, \mathrm{exp}(-r^2/2\lambda^2)$', fontsize=22)

ax.set_xlim([0, Rmax])
ax.legend(loc=1, ncol=2, fontsize=12)
ax.set_ylabel(r'$\frac{c(r,t)}{c_0}$      ', fontsize=32, rotation=0)
ax.set_xlabel('radius [μm]', fontsize=20)

ax.set_ylim([-0.01, 1.02])

fig.tight_layout()
plt.savefig('../../figures/frap_gaussian_concentration_before_recovery.pdf',
        bbox_inches='tight', background_color='white')
# %%
