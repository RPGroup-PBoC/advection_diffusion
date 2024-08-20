import os

import pandas as pd
# Disables false positive warning from use of df.loc value assignment
pd.options.mode.chained_assignment = None

from tqdm import tqdm
tqdm.pandas()
import active_matter_pkg as amp
import numpy as np
from skimage import measure, filters

def comsol_txtparse(fileroot, variable = 'c (mol/m^3)'):
    df_compiled = pd.DataFrame()

    df_header = pd.read_csv(fileroot, skiprows=8, sep=';')

    df_data = pd.read_csv(fileroot, skiprows=8, sep=';')
    df_data = df_data.dropna(axis=1, how='all')

    column_dict = {data:header for data,header in zip(df_data.columns[:len(df_header.columns)],df_header.columns)}
    df = df_data.rename(columns=column_dict)

    # Need to fix headers. Reorganize so parameters and time have
    # their own columns
    df = df.rename(columns={df.columns[0]:'X', df.columns[1]:'Y'})
    df_new = df.iloc[:,0:3]

    column_name = df_new.columns[2]
    t_start = column_name.find('t=') + len('t=')
    t_end = column_name.find(',')
    try:
        df_new.loc[:,'t'] = float(column_name[t_start:t_end])
    except:
        use_filename = True
        df_new.loc[:,'t'] = float(column_name[t_start:])

        # Read filename instead
        _, filename = os.path.split(fileroot)

        D_start = filename.find('D_') + len('D_')
        D_end = filename.find('_alpha')
        D = float(filename[D_start:D_end].replace('p','.'))

        alpha_start = filename.find('alpha_') + len('alpha_')
        alpha_end = filename.find('.txt')
        alpha = float(filename[alpha_start:alpha_end].replace('p','.'))
    
    else:
        use_filename = False
        D_start = column_name.find('D=') + len('D=')
        D_end = D_start + column_name[D_start:].find(',')

        alpha_start = column_name.find('alpha=') + len('alpha=')

        D = float(column_name[D_start:D_end])
        alpha = float(column_name[alpha_start:])

    df_new = df_new.rename(columns={df_new.columns[2]:variable})

    for n in range(3,int(len(df.columns))):
        _df = df.iloc[:,:2]
        _df.loc[:,variable] = df.iloc[:,n]

        column_name = df.columns[n]
        t_start = column_name.find('t=') + len('t=')
        t_end = column_name.find(',')

        if use_filename:
            _df.loc[:,'t'] = float(column_name[t_start:])
        else:
            _df.loc[:,'t'] = float(column_name[t_start:t_end])

        df_new = pd.concat([df_new,_df], ignore_index=True)
    
    df_new['D (um^2/s)'] = D
    df_new['alpha (1/s)'] = alpha
    df_new = df_new.dropna()

    df_compiled = pd.concat([df_compiled, df_new], ignore_index=True)
    
    return df_compiled

def find_halfwidth(df, floor=0.5, smoothing=False, variable='c (mol/m^3)', x_name='X', y_name='Y', time_name='t'):
    """
    Computes the full width of the concentration profile at half maximum
    """

    nonparams = [variable, x_name, y_name, time_name]
    param_names = [param for param in df.columns if (param not in nonparams) and ('Unnamed' not in param)]

    for param in param_names:

        if len(df[param].unique()) > 1:
            print(param, ' has multiple values. Ensure all data of same parameter combination as input')
        
            return
        
    df_hw = pd.DataFrame()

    for t,d in df.groupby(time_name):

        conc_field = amp.image_processing.create_2dfield(d, variable=variable,
                                                         x_name=x_name, y_name=y_name)
        
        # Find fullwidth, half max
        peak = np.max(conc_field)

        im_binary = (conc_field >= (floor * conc_field)) * 1

        regprops = measure.regionprops_table(im_binary,
                                             properties=('area', 'axis_major_length',
                                                         'axis_minor_length', 
                                                         'eccentricity'))
        
        df_regprops = pd.DataFrame(regprops)

        df_regprops['time'] = t

        df_hw = pd.concat([df_hw, df_regprops], ignore_index=True)

    for param in param_names:
        df_hw[param] = df[param].values[0]    
    
    return df_hw

def analyze_comsolsimple(df, smoothing=True, variable='c (mol/m^3)', x_name='X', y_name='Y', time_name='t', frac=0.999):
    """
    Analyzes Comsol data which is presented as a DataFrame

    Inputs :
    ----------
    df : DataFrame, provides variable and position

    intensity_thresh : float, thresholding intensity
    
    variable : str, variable of interest, default 'c (mol/m^3)'
    x_name : str, x-axis variable name, default 'X'
    y_name : str, y-axis variable name, default 'Y'
    time_name : str, time variable name, default 't'

    new_thresholding: bool, determines whether the thresholding should be
                            uniquely obtained for each time point (True) 
                            or the same as the threshold from t=0
    """

    nonparams = [variable, x_name, y_name, time_name]
    param_names = [param for param in df.columns if (param not in nonparams) and ('Unnamed' not in param)]

    for param in param_names:

        if len(df[param].unique()) > 1:
            print(param, ' has multiple values. Ensure all data of same parameter combination as input')
        
            return

    df_analyzing = pd.DataFrame()

    # Analyze unit cells at each time
    for t,d in df.groupby(time_name):

        conc_field = amp.image_processing.create_2dfield(d, variable=variable, 
                                                        x_name=x_name, y_name=y_name)

        # Apply Gaussian filters
        if smoothing:
            im_gauss = filters.gaussian(conc_field, sigma=5)
            thresh = filters.threshold_otsu(im_gauss)
            im_binary = (im_gauss > thresh)

        else:
            thresh = filters.threshold_otsu(conc_field)
            im_binary = (conc_field > thresh)
            im_gauss = conc_field
            
        im_thresholded = im_binary * im_gauss
        initial_counts = np.sum(im_thresholded)

        if t == df['t'].min():
            default_counts = frac * initial_counts

        sort_list = np.sort(im_thresholded, axis=None)
        cumulative = np.cumsum(sort_list[::-1])

        # Compute area from cumulative
        area = np.nonzero(cumulative > default_counts)[0][0]
        final_counts = cumulative[area]

        _df = pd.DataFrame([[area, final_counts, t]],
                            columns=('area','total_counts','time'))

        df_analyzing = pd.concat([df_analyzing, _df])
    
    for param in param_names:
        df_analyzing[param] = df[param].values[0]

    return df_analyzing

def analyze_simplestart(df, noise_floor=1e-15, variable='c (mol/m^3)'):

    area = len(df[df[variable] > noise_floor][variable])
    total_counts = np.sum(df[df[variable] > noise_floor][variable])

    return total_counts, area

def analyze_simplelate(df, starting_counts, frac = 0.9999, variable='c (mol/m^3)'):

    # order the concentrations and take the cumulative sum
    cum_sum = np.cumsum(np.sort(df[variable])[::-1])
    if cum_sum[-1] < frac * starting_counts:
        area = len(cum_sum)
        total_counts = cum_sum[-1]
    else:
        area = np.nonzero(cum_sum >= frac * starting_counts)[0][0]
        total_counts = cum_sum[area]

    return total_counts, area

def analyze_simpleall(df, frac = 0.9999, noise_start = 1e-3, noise_end = 1e-4, parameters = ['D (um^2/s)', 'alpha (1/s)']):
    mintime = df['t'].min()

    for param in parameters:
        if len(df[param].unique()) > 1:
            ValueError('Too many parameter values used here')
            return

    df_compiled = pd.DataFrame()

    noise_list = np.logspace(np.log10(noise_start), np.log10(noise_end), len(df['t'].unique()))
    noise_dict = dict(zip(df['t'].unique(), noise_list))

    for t,d in df.sort_values(by='t').groupby('t'):

        if t == mintime:
            total_counts, area = analyze_simplestart(d, noise_floor=noise_dict[t])
            starting_counts = total_counts.copy()
        else:
            total_counts, area = analyze_simplelate(d, starting_counts, frac=frac)

        _df = pd.DataFrame([[total_counts, area, t]],
                           columns=['total_counts', 'area', 'time'])
        
        df_compiled = pd.concat([df_compiled, _df], ignore_index=True)

    for param in parameters:
        df_compiled[param] = df[param].values[0]

    return df_compiled

def run_comsolanalysis(fileroot, frac=0.9999, noise_start=1e-3, noise_end=1e-4, variable = 'c (mol/m^3)', parameters = ['D (um^2/s)', 'alpha (1/s)']):
    
    df_array = comsol_txtparse(fileroot, variable = variable)
    
    return analyze_simpleall(df_array, noise_start=noise_start, noise_end=noise_end, frac = frac, parameters = parameters)

def run_comsolsimple(fileroot, frac=0.999, variable = 'c (mol/m^3)', smoothing = True):

    df_array = comsol_txtparse(fileroot, variable = variable)

    df_analysis = pd.DataFrame()

    for _,df in df_array.groupby('t'):
        _df_analysis = analyze_comsolsimple(df, frac=frac, smoothing=smoothing)
        df_analysis = pd.concat([df_analysis, _df_analysis],
                                ignore_index=True)

    return df_analysis

def run_comsolhw(fileroot, variable = 'c (mol/m^3)', floor=0.5):
    
    df_array = comsol_txtparse(fileroot, variable = variable)
    
    return find_halfwidth(df_array, variable = variable, floor=floor)
