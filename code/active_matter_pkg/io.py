"""
This subpackage is designed to tidy up datasets, such as raw image
directories with unnecessary `images`."""

import os
import pandas as pd
import numpy as np
from skimage import io

def rm_false_images(directory, keywords=['False Trigger', 'All Off']):
    """
    Removes false images within the directory based on filename. 
    Standard keywords are False Trigger and All Off. Make sure to
    only remove those filenames that do not contain useful images.
    """
    file_list = os.listdir(directory)
    # Keep metadata
    file_list = [item for item in file_list if item[-4:]=='.tif']
    
    for item in file_list:
        if any(list_item in item for list_item in file_list):
            os.remove(os.path.join(directory, file_list, item))

def find_all_tiffs(root_directory):
    """
    Finds all .tif files in root_directory and lists paths. Does not
    work if .tif files are further embedded in subdirectories within
    root_directory
    """
    tiff_list = []

    for root_path, _, files in os.walk(root_directory):
        for f in files:
            if f.endswith('.tif'):
                tiff_list.append(os.path.join(root_path,f))

    return tiff_list

def tiff_walk(root_directory, parse_channels=True):
    """
    Walks through all subdirectories within directory to obtain .tif
    files. Returns list of .tif files in full path. .tif files should
    be at most two directories deep if 'directory' is the root photobleach
    data root: (1) a subdirectory contains different image acquisitions sessions
    for the same (slide,lane,position) combination but the initial aster
    formation process is a separate image acquisition from the photobleaching
    acquisition session; (2) a subdirectory that divides the .tif types
    into channels and cropped images.
    """
    tiff_list = []

    directory_name = os.path.split(root_directory)[-1]

    original_channels = ['DLPRed','DLPYellow','DLP_Red_trimmed','DLP_Yellow_trimmed','DLPBlue']

    # Check if keyword found in image acquisition file exists in 'directory'. Here,
    # choice of 'intervals' is chosen as keyword
    if 'intervals' not in directory_name:
        subdirectory = [d.name for d in os.scandir(str(root_directory)) if (directory_name in d.name and os.path.isdir(d.path))]
        directory = [d.path for d in os.scandir(str(root_directory)) if (directory_name in d.name and os.path.isdir(d.path))]
    else:
        subdirectory = []
        directory = [root_directory]
    
    for direct in directory:
        tiff_directory = [d.path for d in os.scandir(str(direct)) if d.name in original_channels]
        for d in tiff_directory:
            tiff_list.extend(find_all_tiffs(d))

    if parse_channels:
        mt_imgfiles = np.sort([imfile for imfile in tiff_list if '/DLPRed/' in imfile and 'DLP_Red_000.tif' in imfile])
        mot_imgfiles = np.sort([imfile for imfile in tiff_list if '/DLPYellow/' in imfile and 'DLP_Yellow_000.tif' in imfile])
        mt_trimmed = np.sort([imfile for imfile in tiff_list if 'DLP_Red_trimmed/' in imfile and '.tif' in imfile])
        mot_trimmed = np.sort([imfile for imfile in tiff_list if 'DLP_Yellow_trimmed/' in imfile and '.tif' in imfile])
        return mt_imgfiles, mot_imgfiles, mt_trimmed, mot_trimmed, subdirectory
    else:
        return tiff_list, subdirectory

def parse_filename(data_directory):
    if 'skip' in data_directory:
        num_skipstr = data_directory.find('skip')
        num_uscore = num_skipstr + data_directory[num_skipstr:].find('_')
        num_skip = int(data_directory[num_skipstr+4:num_uscore])
    else:
        num_skip = 0
    num_intervals = data_directory.find('_intervals')
    num_uscore_bf_intervals = data_directory[:num_intervals].rfind('_')
    time_interval = int(data_directory[num_uscore_bf_intervals+1:num_intervals-1])

    if 'ATP' in data_directory:
        num_atp = data_directory.find('_ATP')
        num_uscore_bf_atp = data_directory[:num_atp].rfind('_')
        atp_str = data_directory[num_uscore_bf_atp + 1:num_atp]

        # Convert ATP concentration to uM
        if 'mM' in atp_str:
            atp_conc = int(atp_str[:atp_str.find('mM')]) * 1000
        elif 'uM' in atp_str:
            atp_conc = float(atp_str[:atp_str.find('uM')].replace('p','.'))
    else:
        atp_conc = 1400

    if 'pluronic' in data_directory:
        if 'nopluronic' in data_directory:
            pluronic = 0
        elif 'singlepluronic' in data_directory:
            pluronic = 1
        elif 'doublepluronic' in data_directory:
            pluronic = 2
        elif 'xpluronic' in data_directory:
            plur_end = data_directory.find('xpluronic')
            plur_start = data_directory[:plur_end].rfind('_') + len('_')
            pluronic = float(data_directory[plur_start:plur_end].replace('p','.'))
    else:
        pluronic = 1

    if 'frame' in data_directory:
        num_frame = data_directory.find('frame')
        num_uscore = num_frame + data_directory[num_frame:].find('_')
        # Photobleaching occurs prior to the activation cycle listed as frame## in the filename
        # Then there is indexing by 0 in python, thus subtracting by 2
        num_pb = (num_skip + 1) * (int(data_directory[num_frame+5:num_uscore]) - 2) + 1
    else:
        num_pb = np.nan
    df = pd.DataFrame([[num_skip, time_interval, num_pb, atp_conc, pluronic]],
                    columns=['skip number','time interval (s)', 'photobleach frame number', 'ATP (uM)', 'pluronic'])
    
    return df

def underscore_channel(channel):
    if 'DLP' in channel:
        newname = 'DLP_' + channel[3:]
    return newname

def split_tiffstack(root,channels,keep_blue=False):
    """
    If acquired images are stored as a TIF stack, break up according
    to the channels of interest.
    ======================
    root : str, root directory
    channels : str or list containing strings in their order
    keep_blue : bool, option to keep the DLP Blue stack
    """
    directory = os.listdir(root)

    for dataset in directory:
        if dataset == '.DS_Store':
            continue

        path_to_dataset = os.path.join(root, dataset)

        if type(channels) == str:

            if channels not in os.listdir(path_to_dataset):
                os.mkdir(os.path.join(path_to_dataset,channels))

            if any('.ome.tif' in filename for filename in os.listdir(path_to_dataset)):
                imstack = io.imread(os.path.join(path_to_dataset, dataset + '_MMStack_Pos0.ome.tif'))
            else:
                continue

            for n in range(np.shape(imstack)[0]):
                io.imsave(os.path.join(path_to_dataset, channels, 'img_%09d_%s_000.tif' %(n,underscore_channel(channels))),
                                imstack[n, 0, :, :])

        elif type(channels) == list:
            for l in range(len(channels)):
                channel = channels[l]

                if (channel == 'DLPBlue') and (keep_blue):
                    continue

                if channel not in os.listdir(path_to_dataset):
                    os.mkdir(os.path.join(path_to_dataset,channel))

                if any('.ome.tif' in filename for filename in os.listdir(path_to_dataset)):
                    imstack = io.imread(os.path.join(path_to_dataset, dataset + '_MMStack_Pos0.ome.tif'))
                else:
                    continue

                for n in range(np.shape(imstack)[0]):
                    io.imsave(os.path.join(path_to_dataset, channel, 'img_%09d_%s_000.tif' %(n,underscore_channel(channel))),
                                    imstack[n, l, :, :])

    return

def remove_dlpblue(root):
    """
    Removes all DLP Blue channel images in root
    ========
    Input
    root : root directory
    """

    fileset = find_all_tiffs(root)

    alloff_set = [f for f in fileset if 'DLP_Blue_000.tif' in f]
    for f in alloff_set:
        os.remove(f)

    return

def organize_channels(root):
    """
    Organizes channels from possible list of channel names in root
    """
    folders_in_root = os.listdir(root)
    folders_in_root = [f for f in folders_in_root if '.DS_Store' not in f]

    channels = ['DLPBlue', 'DLPRed', 'DLPYellow', 'FRETRed', 'FRETYFP', 'FRETRedCam', 'FRETYFPCam']
    fileends = ['DLP_Blue_000.tif', 'DLP_Red_000.tif', 'DLP_Yellow_000.tif', 'DLP_YFP_000.tif', 'DLP_FRETred_000.tif', 'DLP_FRET-RedCam_000.tif', 'DLP_FRET-YFPCam_000.tif']
    dict_channel = {f:c for f,c in zip(fileends, channels)}

    for folders in folders_in_root:
        folder_path = os.path.join(root,folders)
        confirmed_channels = []
        try:
            files = os.listdir(os.path.join(folder_path,'Pos0'))
        except:
            continue

        for f in files:
            # find index of filename ending in fileends that matches in f
            idxlist = [idx for idx, fe in enumerate(fileends) if fe in f]

            if len(idxlist) == 1:

                idx = idxlist[0]

                if dict_channel[fileends[idx]] not in confirmed_channels:
                    os.mkdir(os.path.join(folder_path,dict_channel[fileends[idx]]))
                    confirmed_channels += [dict_channel[fileends[idx]]]

                os.rename(os.path.join(folder_path,'Pos0',f), os.path.join(folder_path,dict_channel[fileends[idx]],f))
    return

def identify_root(file):
    """
    Determines file path from name of file

    Inputs :
    ----------
    file : str, name of file

    Returns :
    ----------
    data_root : str, path to data
    motortype : str, type of motor
    """

    root_start = '../../../data/active_stress/'
    
    root_ncd = 'photobleach_data'
    root_281 = 'ncd281_photobleach'
    root_308 = 'ncd308_photobleach'
    root_k401 = 'k401_photobleach'
    root_k401bac = 'k401bac_photobleach'
    root_atp = 'atpdilutions'
    root_k401pluronic = 'k401bac_pluronic'
    root_ncdpluronic = 'ncd_pluronic'

    if 'ncd281' in file:
        motortype = 'ncd281'
        root = os.path.join(root_start, root_281)
    elif 'ncd308' in file:
        motortype = 'ncd308'
        root = os.path.join(root_start, root_308)
    elif ('k401bac' in file) or ('K401bac' in file):
        motortype = 'k401bac'
        if 'pluronic' in file:
            root = os.path.join(root_start, root_k401pluronic)
        else:
            root = os.path.join(root_start, root_k401bac)
    elif 'k401bac' not in file and 'k401' in file:
        motortype='k401'
        root = os.path.join(root_start, root_k401)
    elif ('ATP' in file) and ('ncd236' in file):
        motortype = 'ncd236'
        root = os.path.join(root_start, root_atp)
    elif ('pluronic' in file) and ('ncd236' in file):
        motortype = 'ncd236'
        root = os.path.join(root_start, root_ncdpluronic)
    else:
        motortype = 'ncd236'
        root = os.path.join(root_start, root_ncd)
    data_directory = np.sort([os.path.join(root,directory) for directory in os.listdir(root) if 'slide' in directory])
    data_root = str([directory for directory in data_directory if file in directory][0])
    
    return data_root, motortype