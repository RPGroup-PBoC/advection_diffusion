#%%
import sys
sys.path.insert(0,'../')
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocess import Pool, cpu_count
import active_matter_pkg as amp
amp.viz.plotting_style()
tqdm.pandas()

root = '../../../code/comsol/advection_diffusion'

comsol_files = [f for f in os.listdir(root) if 'advdiff_D_' in f]

alpha_list = np.unique([f[f.find('_alpha'):] for f in comsol_files])

comsol_filelist = [os.path.join(root, comsol_file) for comsol_file in comsol_files]

parameter = 'D (um^2/s)'
variable = 'c (mol/m^3)'

def apply_parallel(filelist, func):
    with Pool(cpu_count() - 1) as p:
        ret_list = list(tqdm(p.imap(func, [item for item in filelist]), total=len(filelist)))

    return pd.concat(ret_list, ignore_index=True)

#%%
#bins = [0.999, 0.9999, 1.0]
#for b in bins:

def run_comsolanalysis(fileroot):
    return amp.comsol_analysis.run_comsolanalysis(fileroot, frac=0.999, noise_start=5e-4, noise_end=5e-4)

for alpha in alpha_list:
    comsol_filelist = [os.path.join(root, comsol_file) for comsol_file in comsol_files if alpha in comsol_file]

    if __name__ == '__main__':
        df_compiled = apply_parallel(comsol_filelist, run_comsolanalysis)

    if ('comsol_unitcell' + alpha) in os.listdir('../../analyzed_data/'):
        df_prewritten = pd.read_csv(os.path.join('../../analyzed_data/', 'comsol_unitcell' + alpha), sep=',')
        df_compiled = pd.concat([df_prewritten, df_compiled], ignore_index=True)
        df_compiled.drop_duplicates(subset=['alpha (1/s)', 'D (um^2/s)', 'time'], keep='first', inplace=True, ignore_index=True)
    
    df_compiled.to_csv(os.path.join('../../analyzed_data/','comsol_unitcell_v2' + alpha),
                        sep=',')
    
    #df_compiled = pd.concat([df_compiled, _df_compiled])
#%%
comsol_analyzedlist = [f for f in os.listdir('../../analyzed_data') if 'comsol_unitcell_v2_alpha' in f]

if 'comsol_unitcell_compiled.csv' in os.listdir('../../analyzed_data'):
    df_compiled = pd.read_csv('../../analyzed_data/comsol_unitcell_compiled_v2.csv', sep=',')
else:
    df_compiled = pd.DataFrame()
#%%
for f in comsol_analyzedlist:
    _df = pd.read_csv(os.path.join('../../analyzed_data', f), sep=',')

    #if len(df_compiled) > 0:
    #    if len(df_compiled[df_compiled['alpha (1/s)']==_df['alpha (1/s)'].values[0]]) > 0:
    #        continue

    df_compiled = pd.concat([df_compiled, _df], ignore_index=True)

df_compiled = df_compiled.loc[:, ~df_compiled.columns.str.contains('Unnamed')]
df_compiled.to_csv('../../analyzed_data/comsol_unitcell_compiled_v2.csv')

# %%
