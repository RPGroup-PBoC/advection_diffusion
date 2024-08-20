"""
The image_processing package is designed to process images for identifying
microtubules (in TIRF) or for observing aster formation.
"""
import os
import numpy as np
from skimage import filters, morphology, measure, transform, feature, segmentation, io
from scipy import optimize, spatial
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd
import active_matter_pkg as amp

def filter_mts(image, block_size=5, mask_size=5, yen=False):
    image_norm = (image - image.min()) / (image.max() - image.min())

    thresh_niblack = filters.threshold_niblack(image_norm, window_size=block_size,
                                                k=0.001)

    # Rather than applying the threshold to the image to create a binary
    # image, the threshold array thresh_niblack thickens the MTs, reducing
    # filament break-up. This is used then in the Otsu thresholding to
    # produce the binary image.
    thresh_otsu = filters.threshold_otsu(thresh_niblack)
    im_thresh = (thresh_niblack > thresh_otsu)

    mask = morphology.square(mask_size)
    im_closed = morphology.closing(im_thresh, footprint=mask)

    if yen==True:
        im_subt = image - im_closed
        im_yen = filters.threshold_yen(im_subt)
        im_filtered = im_subt > im_yen
    else:
        im_filtered = im_closed.copy()

    return im_filtered

def border_clear(im_label, edge=3):
    # Remove objects too close to the border
    im_border = np.copy(im_label)

    border = np.ones(np.shape(im_label))
    border[edge:-1*edge,edge:-1*edge] -= 1

    for n in np.unique(im_border):
        if n==0:
            continue
        if np.any(border * [im_border==n]):
            im_border[im_border==n] = 0

    return im_border

def determine_count_nums(im_label):
    """
    Obtains maximum number of objects in the labeled image. Used to determine
    if background subtraction and thresholding must be performed on top of Niblack
    thresholding scheme.
    """
    unique, counts = np.unique(im_label, return_counts=True)

    return unique, counts

def remove_small(im_label, area_thresh=10):
    im_sized = np.copy(im_label)

    unique, counts = determine_count_nums(im_label)

    # Create dictionary except for 0 (background)
    dict_area = dict(zip(unique,counts))

    for label in unique:
        if label > 0 and dict_area[label]<=area_thresh:
            im_sized[im_sized==label] = 0
    
    return im_sized

def remove_large(im_label, area_thresh=10):
    im_sized = np.copy(im_label)

    unique, counts = determine_count_nums(im_label)

    # Create dictionary except for 0 (background)
    dict_area = dict(zip(unique,counts))

    for label in unique:
        if label > 0 and dict_area[label]>=area_thresh:
            im_sized[im_sized==label] = 0
    
    return im_sized

def remove_circulars(im_label, eccen_thresh=0.8):
    im_eccen = im_label.copy()

    im_props = measure.regionprops_table(im_eccen,
                                        properties=['label','eccentricity'])
    df = pd.DataFrame(im_props)

    for n in np.unique(im_eccen):
        if df[df['label']==n]['eccentricity'].values < eccen_thresh:
            im_eccen[im_eccen==n] = 0

    return im_eccen

def are2lines(mt_segment, min_dist=9, min_angle=75):
    """
    Determine if putative microtubules are two microtubules. Uses
    Hough straight lines to determine if there are at least 2
    lines that can be drawn from the putative filament.
    
    Input
    -------
    mt_segment : (M, N), ndarray; cropped region about the putative
                 microtubule
    min_angle : int, minimum angle (in degrees) separating lines (default 75)
    
    Return
    -------
    len(angles)==2 : bool, determines whether there is a crossover
    """
    test_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = transform.hough_line(mt_segment, theta=test_angles)

    _, angles, _ = transform.hough_line_peaks(h, theta, d, 
                                                min_distance=min_dist,
                                                min_angle=min_angle,
                                                num_peaks=2)

    return len(angles)==2

def remove_line_crossovers(image, min_dist=9, min_angle=75, padding=3):
    """
    Removes microtubules that cross over in the images. 
    Input
    -------
    image : (M, N), ndarray; image from which MT crossovers are removed
    min_angle : int, minimum angle (in degrees) separating lines (default 30)
    padding : int, padding around cropped MT segments

    Return
    -------
    im_cleaned : (M, N), ndarray; image with MT crossovers removed
    """
    im_cleaned = image.copy()

    for n in np.unique(image)[1:]:
        x,y = np.where(image==n)
        mt_segment = image[x.min()-padding:x.max()+padding,y.min()-padding:y.max()+padding]
        
        if are2lines(mt_segment, min_dist=min_dist, min_angle=min_angle):
            im_cleaned = np.where(im_cleaned==n, 0, im_cleaned)

    return im_cleaned

def process_mt_images(image, block_size=3, mask_size=3, count_thresh=200, edge=3, area_thresh=10, min_dist=9, min_angle=75, padding=3):
    im_filtered = filter_mts(image, block_size=block_size, mask_size=mask_size)
    im_label, n_labels = measure.label(im_filtered, return_num=True)
    # Determine if Yen thresholding background subtraction is necessary
    unique, _ = determine_count_nums(im_label)
    if unique[-1] > count_thresh:
        im_filtered = filter_mts(image, block_size=block_size, mask_size=mask_size, yen=True)
        im_label, n_labels = measure.label(im_filtered, return_num=True)
    im_internal = border_clear(im_label, edge=edge)
    im_sized = remove_small(im_internal, area_thresh=area_thresh)
    im_thinned = morphology.thin(im_sized)
    im_relabel = measure.label(im_thinned)
    im_noxovers = remove_line_crossovers(im_relabel, min_dist=min_dist,
                                        min_angle=min_angle, padding=padding)

    return im_noxovers

def gaussian_2d(coords, mu_x, mu_y, sigma_x, sigma_y, coeff, offset=0):
    """
    Defines a 2D Gaussian function

    Input
    -------
    coords : 2xN array; x and y coordinates along the Gaussian
    mu_x, mu_y : float; x and y coordinates of Gaussian peak
    sigma_x, sigma_y : float; standard deviations of Gaussian
    coeff : float; coefficient in front of the exponential term
    offset : float; offset value if Gaussian doesn't go to 0. Default 0

    Return
    -------
    coeff * np.exp(-(x - mu_x)**2/(2*sigma_x**2) - (y - mu_y)**2/(2*sigma_y**2))
    """
    exponent = - (coords[:,0] - mu_x)**2/(2*sigma_x**2) - (coords[:,1] - mu_y)**2/(2*sigma_y**2)
    return offset + coeff * np.exp(exponent)

def gaussian_fit(image, alpha_guess, sigma_xguess=20, sigma_yguess=20):
    y = np.arange(np.shape(image)[1], step=1)
    x = np.arange(np.shape(image)[0], step=1)
    Y, X = np.meshgrid(y,x)
    coords = np.column_stack((np.ravel(X), np.ravel(Y)))

    p0 = [np.shape(image)[0]/2, np.shape(image)[1]/2, sigma_xguess, sigma_yguess, alpha_guess]
    popt, pcov = optimize.curve_fit(gaussian_2d, coords, np.ravel(image), p0=p0)

    return popt, pcov

def process_aster_cells(image, sigma=100, small_area_thresh=20, area_thresh=20):
    """
    Performs post processing on post-photobleached cells during aster 
    network formation. Cleans background with Gaussian blurring before
    applying mean thresholding and removing small objects.

    Input
    -------
    image : (M, N), ndarray; raw image to be processed
    sigma : float; sigma on Gaussian filter, default is 100 pixels
    small_area_thresh : int; threshold number of pixels to be large enough to keep
    area_thresh : int; second area thresholding value based on regionprops

    Return
    -------
    im_relabel : (M, N), ndarray; image composed of integer values for each aster unit cell
    regprops : list of RegionProperties
    """
    im_blur = filters.gaussian(image, sigma=sigma)
    im_subt = image - (2**16 - 1) * im_blur
    im_subt[im_subt < 0] = 0

    thresh_mean = filters.threshold_mean(im_subt)
    im_binary = (im_subt > thresh_mean)
    im_binary = morphology.remove_small_objects(im_binary, 20)

    # Try on mean thresholding scheme
    im_labels, num_labels = measure.label(im_binary, return_num=True)
    regionprops = measure.regionprops_table(im_labels, image,
                                            properties=('area','bbox','centroid',
                                                        'convex_area','convex_image',
                                                        'coords','eccentricity',
                                                        'equivalent_diameter','euler_number',
                                                        'extent','filled_area','filled_image',
                                                        'image','inertia_tensor','inertia_tensor_eigvals',
                                                        'intensity_image','label','local_centroid',
                                                        'major_axis_length','max_intensity','mean_intensity',
                                                        'min_intensity','minor_axis_length','moments',
                                                        'moments_central','moments_hu','moments_normalized',
                                                        'orientation','perimeter',
                                                        'slice','solidity','weighted_centroid',
                                                        'weighted_local_centroid','weighted_moments',
                                                        'weighted_moments_central','weighted_moments_hu',
                                                        'weighted_moments_normalized'))
    df_regionprops = pd.DataFrame(regionprops)

    for index in range(1,num_labels+1):
        if df_regionprops[df_regionprops['label']==index]['area'].values[0] <= area_thresh:
            im_labels[im_labels==index] = 0

    im_relabel = measure.label(im_labels)
    regprops = measure.regionprops_table(im_relabel, image,
                                            properties=('area','bbox','centroid',
                                                        'convex_area','convex_image',
                                                        'coords','eccentricity',
                                                        'equivalent_diameter','euler_number',
                                                        'extent','filled_area','filled_image',
                                                        'image','inertia_tensor','inertia_tensor_eigvals',
                                                        'intensity_image','label','local_centroid',
                                                        'major_axis_length','max_intensity','mean_intensity',
                                                        'min_intensity','minor_axis_length','moments',
                                                        'moments_central','moments_hu','moments_normalized',
                                                        'orientation','perimeter',
                                                        'slice','solidity','weighted_centroid',
                                                        'weighted_local_centroid','weighted_moments',
                                                        'weighted_moments_central','weighted_moments_hu',
                                                        'weighted_moments_normalized'))
    df_regprops = pd.DataFrame(regprops)

    return im_relabel, df_regprops

def normalize(im):
    # Sets renormalization of values in NxM array to be between 0 and 1
    return (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))

def image_mask(image, sigma=30, hw=8):
    im_bk = filters.gaussian(image, sigma=sigma) * (2**16-1)
    im_subt = image - im_bk
    im_subt[im_subt<0] = 0

    thresh_mean = filters.threshold_mean(im_subt)
    im_thresh = (im_subt > thresh_mean)

    # thresholding background image to help remove off-target areas
    thresh_bk = filters.threshold_mean(im_bk)
    im_bk_thresh = (im_bk > thresh_bk)
    im_thresh = im_bk_thresh * im_thresh

    im_binary = morphology.remove_small_objects(im_thresh)
    im_label, n_label = measure.label(im_binary, return_num=True)
    im_border = border_clear(im_label,edge=hw)
    return (im_border>0)

def crop_flow_field(image, u, v, hw=8, sigma=30):

    x = np.arange(0, np.shape(image)[0], 1)
    y = np.arange(0, np.shape(image)[1], 1)

    Y_im, X_im = np.meshgrid(y,x)
    im_mask = image_mask(image, sigma=sigma, hw=hw)

    for n_x in range(image.shape[0]):
        for n_y in range(image.shape[1]):
            x_cent = X_im[n_x,n_y]
            y_cent = Y_im[n_x,n_y]

            x_low = max(0, int(x_cent-hw))
            x_high = min(int(x_cent+hw), image.shape[0])
            y_low = max(0,int(y_cent-hw))
            y_high = min(int(y_cent+hw), image.shape[1])

            window = np.s_[x_low:x_high,y_low:y_high]
            if 1 not in im_mask[window]:
                u[n_x,n_y] = np.nan
                v[n_x,n_y] = np.nan

    return u, v

def mask_value(M, im_mask, num=False):
    if np.shape(M) != np.shape(im_mask):
        raise ValueError('input array and mask image are not the same shape')
    
    M_masked = M.copy()

    if num:
        M_masked[im_mask==0] = 0
    else:
        M_masked[im_mask==0] = np.nan

    return M_masked

def create_window(x,y,winsize):
    """
    Creates a new image window to crop a larger original image
    
    Inputs :
    -----------
    x,y : floats, centers of the image

    Returns :
    -----------
    winsize : float, half-width of window
    """
    return np.s_[int(y-winsize):int(y+winsize),int(x-winsize):int(x+winsize)]

def clean_unitcells(im_binary, small_thresh = 1000, large_thresh = 3000):
    """
    Cleans up images containing unit cells by removing large
    objects, small objects, objects cut off at the boundary
    of the image, and closing any small holes in the unit
    cells from the thresholding.

    Inputs :
    ------------
    im_binary : M x N array, binary representation of segmented image

    small_thresh : int, default 1000 pixels, area thresholding below which object is considered
                    too small to be unit cell
    large_thresh : int, default 3000 pixels, area thresholding above which object is considered
                    too large to be unit cell

    Returns :
    ------------
    im_filled : M x N array, integer representation denoting cleaned, labeled unit cells
    """
    im_label = measure.label(im_binary)

    # Remove smalls and objects near borders
    im_border = border_clear(im_label, edge=5)
    im_small = remove_small(im_border, area_thresh = small_thresh)
    im_justright = remove_large(im_small, area_thresh = large_thresh)
    im_bw = (im_justright > 0)

    im_relabel, n_labels = measure.label(im_bw, return_num=True)

    im_filled = np.zeros(np.shape(im_binary))
    im_mask = np.zeros(np.shape(im_binary))
    square = morphology.square(3)

    for m in range(1,n_labels+1):
        im_nclosed = morphology.closing(im_relabel == m, footprint = square)
        
        edges = feature.canny(im_nclosed)
        edges_closed = morphology.closing(edges, footprint = square)
        filled_cell = ndi.binary_fill_holes(edges_closed)
        im_filled += filled_cell * m
        im_mask += filled_cell

    idx_overlap = np.argwhere(im_mask > 1)

    for idx in idx_overlap:
        im_filled[idx[0],idx[1]] = 0

    return im_filled

def match_cellid(df_current, df_previous, id_max,
                dist_thresh=20, area_thresh=3000):
    """
    Introduces new row to the df_current DataFrame using cellID numbers
    as defined during the t=0 analysis. Must contain labels, areas, and 
    centroids. If distance between unit cell and putative matching cell
    of previous timepoint is at least dist_thresh pixels away, the unit
    cell is considered newly identified. If size of unit cell is too 
    large, it is rejected.

    Inputs :
    -----------
    df_current : DataFrame, current dataframe of identified unit cells
    df_previous : DataFrame, measured properties of unit cells at previous time points
    
    id_max : int, maximum ID number

    dist_thresh : float, maximum distance between putative matched cells
    area_thresh : float, maximum size of putative cell
    dx : float, x-displacement of the network center
    dy : float, y-displacement of the network center

    Returns :
    ------------
    df_current : DataFrame, contains new column for matched cell ID numbers
    id_max : int, new max cell ID number
    """
    # Pull df of previous analyzed frame
    df_timepoint = df_previous[(df_previous['time'] == df_previous['time'].max())]
    #df_timepoint.loc[:,'centroid-0'] += dy
    #df_timepoint.loc[:,'centroid-1'] += dx

    dist = spatial.distance.cdist(df_current[['centroid-0','centroid-1']],
                                    df_timepoint[['centroid-0','centroid-1']])

    id_idx = [np.argwhere(dist[i,:]==dist[i,:].min())[0][0] for i in range(len(dist[:,0]))]
    id = df_timepoint['cellID'].values[id_idx]

    for _id in range(len(id)):
        if dist[_id,id_idx[_id]] > dist_thresh:
            id[_id] = id_max
            id_max += 1

        if df_current[df_current['label']==_id+1]['area'].values[0] > area_thresh:
            id[_id] = -1

    df_current['cellID'] = id

    return df_current, id_max

def add_layer(im_ucmask, im, thresholding=False):
    """
    Adds a layer of pixels around the unit cells for when the unit cell
    intensity total pixel intensity is below the threshold intensity. If
    thresholding == True, uses Otsu thresholding to add a subset of the
    layer.
    
    Inputs :
    ----------
    im_ucmask : M x N array, binary of unit cell
    im : M x N array, image field with intensities

    Returns :
    ----------
    im_updatedmask : M x N array, binary of unit cell with new layer added
    im_layer : M x N array, intensity image of additional layer (post thresholding)
    intensity_total : float, updated total intensity
    """
    im_pad = segmentation.expand_labels(im_ucmask, distance=1)

    im_boundary = segmentation.find_boundaries(im_pad, mode='inner')
    im_boundIntensity = im_boundary * im

    if thresholding:
        bound_thresh = filters.threshold_otsu(im_boundIntensity[im_boundary > 0])
        im_layer = (im_boundIntensity > bound_thresh) * 1

        im_updatedmask = im_ucmask + im_layer
        intensity_total = np.sum(im_updatedmask * im)
    else:
        intensity_total = np.sum(im_pad * im)
        im_layer = im_boundary
        im_updatedmask = im_ucmask + im_boundary

    return im_updatedmask, im_layer, intensity_total

def adjust_intensity(im_ucmask, im, intensity1, intensity2, intensity_thresh,
                    thresholding=False, layer_thresh=np.inf):
    """
    Compares intensity2, the pixel intensity of the unit cell in the current frame,
    to intensity1, the pixel intensity of the unit cell in a previous frame.
    If they differ by a factor of 1 - intensity_thresh, add a layer of pixels to the
    unit cell if more pixels need to be added, or remove individual pixels if pixels
    need to be removed. 

    Inputs :
    ----------
    im_ucmask : M x N array, binary image of unit cell
    im : M x N array, intensity image

    intensity1 : float, total intensity of unit cell at previous time point
    intensity2 : float, total intensity of unit cell at current time point
    intensity_thresh : float, fraction (0,1.0) of intensity that total intensity values
                                can be relative to each other

    thresholding : bool, default False; for adding sublayers if adding pixels
    layer_thresh : int, default infinity; number of layers that can be added before 
                        program terminates

    Returns: 
    ----------
    im_ucmask : M x N array, updated binary image of unit cell
    intensity2 : float, updated total intensity of unit cell at current time point
    n_layers : int, number of layers that were added
    """

    # Have a counter for layers added to image. If it exceeds layer_thresh layers, that cellID is removed
    n_layers = 0

    if intensity2 < intensity_thresh * intensity1:

        while (intensity2 < intensity_thresh * intensity1) and (n_layers < layer_thresh + 1):
            n_layers += 1

            im_ucmask, im_layer, intensity2 = add_layer(im_ucmask, im, thresholding=thresholding)

            if intensity2 > intensity1:

                # Remove dimmest pixels
                im_boundary = im_layer * im

                while (intensity2 > intensity1) and (len(np.nonzero(im_boundary)[0])>0):

                    min_intensity = np.min(im_boundary[np.nonzero(im_boundary)])
                    y_dim, x_dim = np.argwhere(im_boundary==min_intensity)[0]
                    im_ucmask[y_dim,x_dim] = 0
                    im_boundary[y_dim,x_dim] = 0
                    intensity2 -= min_intensity

    elif intensity2 * intensity_thresh > intensity1:

        while intensity2 * intensity_thresh > intensity1:

            im_boundary = segmentation.find_boundaries(im_ucmask, mode='inner') * im

            #if np.sum(im_boundary) < (intensity2 * intensity_thresh - intensity1):
            #    im_ucmask -= segmentation.find_boundaries(im_ucmask, mode='inner')
            #    intensity2 -= np.sum(im_boundary)
            #else:
            min_intensity = np.min(im_boundary[np.nonzero(im_boundary)])
            y_dim,x_dim = np.argwhere(im_boundary==min_intensity)[0]
            im_ucmask[y_dim,x_dim] = 0
            intensity2 -= min_intensity

    return im_ucmask, intensity2, n_layers

def length2pxl_convert(x):
    """
    Converts the absolute length to pixel count

    Inputs :
    ----------
    x : arr, array of x values

    Returns :
    ----------
    x_array : arr, sorted array of x values
    dict_x : dict, dictionary converting x to pxl count
    """

    # Reconfigure to be in an array
    x_array = np.sort(x)
    x_integer = np.arange(0, len(x_array), 1)
    dict_x = {a:i for a,i in zip(x_array, x_integer)}

    return x_array, dict_x

def create_2dfield(df, variable= 'c (mol/m^3)', x_name='X', y_name='Y', no_negatives=True):
    """
    Creates a concentration field from dataframe containing X, Y, variable

    Inputs :
    ----------
    df : DataFrame, contains variable, x, y

    variable : str, label for variable to be put into M x N array
    x_name : str, label for x column
    y_name : str, label for y column

    allow_negatives : bool, correction for negative numbers turning to 0

    Returns :
    -----------
    field : M x N array, field for variable of interest
    """
    x_array, dict_x = length2pxl_convert(df[x_name].unique())
    y_array, dict_y = length2pxl_convert(df[y_name].unique())

    conc_field = np.zeros((len(x_array), len(y_array)))

    for xy, _df in df.groupby([x_name, y_name]):
        x,y = xy
        conc_field[dict_x[x], dict_y[y]] = _df[variable].values[0]

    if no_negatives:
        
        conc_field[conc_field < 0] = 0
        
    return conc_field

def background_subtract(im, sigma=20, bitdepth=16):
    """
    Perform Gaussian blur and subtraction on image
    
    Inputs :
    ----------
    im : M x N array, uint depending on bit depth
    sigma : size of Gaussian blur
    bitdepth : int, bit depth

    Return :
    ----------
    im_norm : M x N array, normalized corrected image
    """

    im_gauss = filters.gaussian(im, sigma=sigma) * (2**bitdepth - 1)
    
    im_subt = im - im_gauss

    return normalize(im_subt)


def initialize_unitcells(field, offset=0, is_image=True, thresh_method='otsu', small_thresh = 1000, large_thresh = 3000):
    """
    Provides initial identification of unit cells in t=0 frame

    Inputs :
    ----------
    field : M x N array, image of field

    is_image : bool, checks if data is image and needs cleaning up
    thresh_method : str, thresholding method to use for initial identification
    small_area : int, pixel size below which unit cell is rejected in clean_unitcells function
    large_area : int, pixel size above which unit cell is rejected in clean_unitcells function

    Returns :
    ----------
    df : dataframe, initial information on unit cells
    id_max : maximum label number
    thresh : float, thresholding value
    merged : arr (0s), list of binary values for merging
    """

    if is_image:
        im = background_subtract(field, sigma=20, bitdepth=16)
    else:
        im = field

    if thresh_method == 'otsu':
        thresh = filters.threshold_otsu(im)

    elif thresh_method == 'mean':
        thresh = filters.threshold_mean(im)

    elif thresh_method == 'triangle':
        thresh = filters.threshold_triangle(im)

    im_binary = (im > thresh)

    im_filled = clean_unitcells(im_binary, small_thresh=small_thresh, large_thresh=large_thresh)

    regprops = measure.regionprops_table(im_filled.astype(int), intensity_image=(field - offset),
                                            properties=['area','centroid','label', 'intensity_image',
                                                        'weighted_centroid','mean_intensity'])

    df = pd.DataFrame(regprops)

    df['total_intensity'] = df['intensity_image'].apply(lambda x : np.sum(x))

    df['cellID'] = df['label']
    df['merged'] = 0

    id_max = df['label'].max() + 1

    merged = np.zeros((len(df), 2))
    merged[:,0] = df['cellID'].values[:]

    regprops_center = measure.regionprops_table((im_filled>0).astype(int), intensity_image=(field - offset),
                                            properties=['weighted_centroid'])
    df_regpropscent = pd.DataFrame(regprops_center)
    y_cent, x_cent = df_regpropscent[['weighted_centroid-0','weighted_centroid-1']].values[0]
    df['x_cent'] = x_cent
    df['y_cent'] = y_cent
    
    df['time'] = 0

    df['total_intensity'] = [np.sum(entry) for entry in df['intensity_image'].values[:]]
    df = df.drop(columns=['intensity_image'])

    return df, id_max, thresh, merged

def later_unitcells(field, df_previous, t, id_max, merged, intensity_thresh, offset = 0, is_image=True, thresh=None, thresh_method=None, layer_thresh=np.inf, small_thresh=1000, large_thresh=3000, thresholding=True):

    x_cent = df_previous['x_cent'].values[0]
    y_cent = df_previous['y_cent'].values[0]

    if is_image:
        im = background_subtract(field, sigma=20, bitdepth=16)
    else:
        im = field

    if thresh == None:
        if thresh_method == 'otsu':
            thresh = filters.threshold_otsu(im)

        elif thresh_method == 'mean':
            thresh = filters.threshold_mean(im)

        elif thresh_method == 'triangle':
            thresh = filters.threshold_triangle(im)
        else:
            print('No thresholding method specified')
            return

    im_binary = (im > thresh)

    im_filled = clean_unitcells(im_binary, small_thresh=small_thresh, large_thresh=large_thresh)

    regprops = measure.regionprops_table(im_filled.astype(int), intensity_image=(field - offset),
                                            properties=['area','centroid','label', 'intensity_image',
                                                        'weighted_centroid','mean_intensity'])
    _df = pd.DataFrame(regprops)
    
    _df['total_intensity'] = _df['intensity_image'].apply(lambda x : np.sum(x))

    _df, id_max = match_cellid(_df, df_previous, id_max, area_thresh=large_thresh)

    im_resizing = np.zeros(np.shape(im))

    # Check total intensity of previous image
    for cellid,_d in _df.groupby('cellID'):

        if (cellid == -1) or ((len(df_previous[(df_previous['cellID']==cellid) & (df_previous['time']==df_previous['time'].min())])==0)):
            continue

        intensity_tot1 = df_previous[(df_previous['cellID']==cellid) & (df_previous['time']==df_previous['time'].min())]['total_intensity'].values[0]
        intensity_tot2 = _d['total_intensity'].values[0]
        if intensity_tot2==0:
            continue
        im_ucmask = (im_filled==_d['label'].values[0]) * 1

        im_ucmask, intensity_tot2, n_layers = adjust_intensity(im_ucmask, field - offset,
                                                                intensity_tot1, intensity_tot2, intensity_thresh,
                                                                thresholding=thresholding, layer_thresh = layer_thresh)

        # Check to see if the unit cell overlaps with one that already exists
        overlap = np.argwhere(im_ucmask * im_resizing > 0)
        if len(overlap)>0:
            # Find label number in nth image
            overlap_id = np.unique([im_resizing[r,c] for r,c in overlap])
            
            # Include the label of the unit cell being processed
            overlap_id = np.append(overlap_id,_d['label'].values[0])

            # Tie back to cell ID
            for _overlap_id in overlap_id:
                _cellid = _df[_df['label']==_overlap_id]['cellID'].values[0]
                try:
                    cellid_idx = np.argwhere(merged[:,0]==_cellid)[0]
                except:
                    print('Overlap not found')
                else:
                    merged[cellid_idx,1] = 1
                    im_resizing[im_resizing==_cellid] = 0
        elif n_layers == layer_thresh + 1:
            try:
                cellid_idx = np.argwhere(merged[:,0]==_d['cellID'].values[0])[0]
            except:
                print('Overlap not found')
            else:
                merged[cellid_idx,1]=1
        else:
            im_resizing += im_ucmask * _d['label'].values[0]
    
    regprops2 = measure.regionprops_table(im_resizing.astype(int), intensity_image=(field - offset),
                                        properties=['area','centroid','label','intensity_image',
                                                    'weighted_centroid','mean_intensity'])
    df_reg2 = pd.DataFrame(regprops2)
    _df = df_reg2.merge(_df[['label','cellID']],on='label')

    df_merged = pd.DataFrame(merged,columns=['cellID','merged'])
    _df = _df.merge(df_merged, on='cellID')

    _df['total_intensity'] = _df['intensity_image'].apply(lambda x : np.sum(x))

    _df = _df.drop(columns=['intensity_image'])

    _df['x_cent'] = x_cent
    _df['y_cent'] = y_cent
    _df['time'] = t

    if len(_df) < len(df_previous[(df_previous['time']==df_previous['time'].min())])/4:
        continue_analysis = 0
    else:
        continue_analysis = 1

    return pd.concat([df_previous, _df]), id_max, merged, continue_analysis

def analyze_comsol(df, intensity_thresh, variable='c (mol/m^3)', x_name='X', y_name='Y', thresh_method='otsu', time_name='t', new_thresholding=False, single_square=False, small_thresh = 500, large_thresh = 3000):
    """
    Analyzes Comsol data which is presented as a DataFrame

    Inputs :
    ----------
    df : DataFrame, provides variable and position

    intensity_thresh : float, thresholding intensity, between 0 and 1
    
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

    # Analyze unit cells at each time
    for t,d in df.groupby(time_name):

        if t == 0:
            continue_analysis = 1

        elif (t > 0) and (continue_analysis == 0):
            break

        conc_field = create_2dfield(d, variable=variable, 
                                    x_name=x_name, y_name=y_name)

        # Check to see if the dataset is merged into only a couple of objects
        thresh = filters.threshold_mean(conc_field)
        im_binary = (conc_field > thresh)

        if not single_square:
            im_label = measure.label(im_binary)

            if len(np.unique(im_label)) < 3:
                print('dataset merged at t = %.2f' %t)
                break

        if t == 0:
            df_analyzing, id_max, thresh, merged = initialize_unitcells(conc_field, is_image=False, thresh_method=thresh_method,
                                                                        small_thresh=small_thresh, large_thresh=large_thresh)

        else:
            if new_thresholding:
                df_analyzing, id_max, merged, continue_analysis = later_unitcells(conc_field, df_analyzing, 
                                                                                t, id_max, merged, intensity_thresh,
                                                                                is_image=False, thresh_method=thresh_method,
                                                                                small_thresh=small_thresh, large_thresh=large_thresh)
            else:
                df_analyzing, id_max, merged, continue_analysis = later_unitcells(conc_field, df_analyzing, 
                                                                                t, id_max, merged, intensity_thresh,
                                                                                is_image=False, thresh=thresh,
                                                                                small_thresh=small_thresh, large_thresh=large_thresh)
            
    for param in param_names:
        df_analyzing[param] = df[param].values[0]

    df_analyzing['radius'] = np.sqrt((df_analyzing['weighted_centroid-1'] - df_analyzing['x_cent'])**2 + (df_analyzing['weighted_centroid-0'] - df_analyzing['y_cent'])**2)

    return df_analyzing
    
def analyze_photobleach(file, offset, n_tot, intensity_thresh, layer_thresh, small_area = 800, large_area = 5000, n_start=0, new_thresholding=False, activation_radius=125, thresh_method='otsu'):
    """
    Perform image segmentation and data collection of photobleached unit cells.
    Analyzes unit cells by ensuring total fluorescence intensity is preserved.

    Inputs :
    -----------
    file : str, name of file, does not contain root
    
    offset : float, offset value for intenisty preservation.
    n_tot : int, number of time frames studied after the initial frame
    intensity_thresh : float, thresholding intensity value
    layer_thresh : int, number of layers to add pixels before termination

    n_start : int, default 0, starting frame number
    new_thresholding : bool, default False, whether to perform new thresholding for
                        successive images

    Returns :
    -----------
    df : DataFrame, contains information of unit cells for given file
    """
    # Loading graticule information
    df_graticule = pd.read_csv('../../analyzed_data/objective_pxl_micron_scale.csv', sep=',')
    um_per_pxl = df_graticule['micron_per_pixel'].values[0]

    # Includes dictionary for determining photobleach number for some data
    numpb_dict = dict({'210426_slide1_lane1_pos1':34,
                    '210426_slide1_lane1_pos2':31,
                    '210426_slide1_lane2_pos1':87})
    
    data_root, motortype = amp.io.identify_root(file)
    _, _, mt_trimmed, _, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

    if len(subdirectory)>0:

        if any('filename_order.csv' in filename for filename in os.listdir(data_root)):

            df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
            data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    
    df_info = amp.io.parse_filename(data_root)
    num_pb = df_info['photobleach frame number'].values[0]
    if np.isnan(num_pb):
        num_pb = numpb_dict[file]

    for n in range(n_start, np.min([len(mt_trimmed), n_start + n_tot])):

        if n == n_start:
            continue_analysis = 1
        elif (n > n_start) and (continue_analysis == 0):
            break

        im = io.imread(mt_trimmed[n])
        
        if thresh_method == 'custom':
            if file=='210519_slide2_lane2_pos1_10s_intervals_10ms_iLidmicroNcd_Alloff_100ms_DLPYellow_100ms_DLPRed_50ms_DLPBlue_skip1_frame30_photobleach_1':
                thresh_method = 'triangle'
            else:
                thresh_method = 'mean'
        else:
            thresh_method = 'otsu'

        if n==n_start:
            df, id_max, thresh, merged = initialize_unitcells(im, offset=offset, 
                                                            thresh_method=thresh_method,
                                                            small_thresh = small_area, large_thresh = large_area)
            
            # Override preset time
            df.loc[:, 'time'] = n

        else:
            if new_thresholding:
                df, id_max, merged, continue_analysis = later_unitcells(im, df, n, id_max,
                                                                        merged, intensity_thresh,
                                                                        thresh_method=thresh_method,
                                                                        layer_thresh=layer_thresh,
                                                                        small_thresh = small_area, 
                                                                        large_thresh = large_area,
                                                                        offset = offset, thresholding=True)
            else:
                df, id_max, merged, continue_analysis = later_unitcells(im, df, n, id_max,
                                                                        merged, intensity_thresh,
                                                                        thresh=thresh,
                                                                        layer_thresh=layer_thresh,
                                                                        small_thresh = small_area, 
                                                                        large_thresh = large_area, 
                                                                        offset = offset, thresholding=True)

    df['time interval (s)'] = df_info['time interval (s)'].values[0]
    df['num_pb'] = num_pb
    df['filename'] = data_root
    df['motor'] = motortype
    df['pluronic'] = df_info['pluronic'].values[0]
    df['ATP (uM)'] = df_info['ATP (uM)'].values[0]
    df['radius'] = np.sqrt((df['weighted_centroid-1'] - df['x_cent'])**2 + (df['weighted_centroid-0'] - df['y_cent'])**2) * um_per_pxl

    df = df[df['radius'] < activation_radius]

    return df

def crop_network(im, x_cent, y_cent, winsize, show_process=False):
    """
    Crops the image from finding the center of the network
    
    Inputs :
    ----------------
    im : MxN array, image to be cropped

    x_cent, y_cent : float, putative x,y centers of image
    winsize : int, intended size of the window

    show_process : bool, default False, shows plots of output

    Returns :
    ----------------
    im_cropped : winsize x winsize array, cropped image
    
    x_newcent, y_newcent : float, new center of network
    """
    
    winsize_expanded = int(winsize * 1.1)

    win_expand = create_window(y_cent, x_cent, winsize_expanded)
    im_win = im[win_expand]

    im_gauss = filters.gaussian(im_win,sigma=30) * (2**16-1)
    im_subt = im_win - im_gauss
    im_norm = normalize(im_subt)
    thresh = filters.threshold_otsu(im_norm)
    im_binary = (im_norm > thresh)
    #square = morphology.square(3)
    #im_closed = morphology.closing(im_binary, selem=square)

    im_label = measure.label(im_binary)
    # Remove smalls and objects near borders
    im_border = border_clear(im_label, edge=5)
    im_small = remove_small(im_border, area_thresh=400)

    im_bw = (im_small > 0)

    im_relabel, n_labels = measure.label(im_bw, return_num=True)

    im_mask = np.zeros(np.shape(im_win))
    square = morphology.square(3)
    for m in range(n_labels+1):
        if m==0:
            continue
        im_nclosed = morphology.closing(im_relabel==m, footprint=square)
        
        edges = feature.canny(im_nclosed)
        edges_closed = morphology.closing(edges, footprint=square)
        filled_cell = ndi.binary_fill_holes(edges_closed)
        im_mask += filled_cell

    regprops_center = measure.regionprops_table((im_mask>0).astype(int), intensity_image=im_win,
                                            properties=['weighted_centroid'])
    df_regpropscent = pd.DataFrame(regprops_center)
    x_newcent, y_newcent = df_regpropscent[['weighted_centroid-0','weighted_centroid-1']].values[0]

    # Create window in the full field of view frame

    y_newcent += y_cent - winsize_expanded
    x_newcent += x_cent - winsize_expanded
    new_win = create_window(y_newcent, x_newcent, winsize)

    if show_process:
        _, ax = plt.subplots(1,2,figsize=(16,8))
        ax[0].imshow(im_bw)
        ax[1].imshow(im[new_win])
        ax[1].scatter(winsize,winsize)
        plt.show()

    return im[new_win], x_newcent, y_newcent

def initial_center(im_prephotobleach, show_process=False):
    """
    Finds initial center of the network and window size using image before photobleaching occurs

    Inputs :
    -----------
    im_prephotobleach : MxN array, uint16, image of contracting network

    show_process : bool, default False, ability to inspect cropped image based on center

    Returns :
    -----------
    x_cent, y_cent : floats, (x,y) center of the network
    winsize : int, recommended window size for cropping the network
    """

    im_gauss = filters.gaussian(im_prephotobleach, sigma=5) * (2**16-1)

    thresh = filters.threshold_yen(im_gauss)
    im_binary = (im_gauss > thresh)

    im_label = measure.label(im_binary)
    im_border = border_clear(im_label, edge=10)
    im_clean = remove_small(im_border, area_thresh=100)

    regprops = measure.regionprops_table(im_clean, intensity_image=im_gauss, 
                                        properties=('weighted_centroid', 
                                                    'centroid', 'area',
                                                    'major_axis_length'))
    df_regprops = pd.DataFrame(regprops)

    # Determine window size and center coordinates. Window size is fixed from here
    if len(df_regprops)<1:
        y_cent, x_cent = np.shape(im_prephotobleach)
        winsize = 500
        y_cent = int(y_cent/2)
        x_cent = int(x_cent/2)
    else:
        x_cent, y_cent = df_regprops[df_regprops['area']==df_regprops['area'].max()][['weighted_centroid-0','weighted_centroid-1']].values[0]
        winsize = int(0.65 * df_regprops[df_regprops['area']==df_regprops['area'].max()]['major_axis_length'].values[0])
        
    if show_process:
        win = create_window(y_cent, x_cent, winsize)
        _, ax = plt.subplots(1,1,figsize=(8,8))
        ax.imshow(im_prephotobleach[win])
        plt.show()

    return x_cent, y_cent, winsize

def crop_imageset(data_root, n_tot=20, show_process=False):
    """
    Crops n_tot images from the image stack starting from num_pb

    Inputs :
    ----------
    data_root : str, root directory of image set
    
    n_tot : int, default 20, number of frames after photobleaching to crop
    show_process : bool, default False, provides plots of cropped images

    Returns :
    -----------
    files : saved in data_root under their 'DLP_Red_trimmed' and for two-color experiments, with 'DLP_Blue_trimmed'
    """
    zerolist = ['210426_slide1_lane1_pos1', '210426_slide1_lane1_pos2', '210426_slide1_lane2_pos1']
    _, _, _, _, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

    if 'mt647_mt488' in data_root:
        files = amp.io.find_all_tiffs(data_root)
        blue_imgfiles = np.sort([f for f in files if '/DLPBlue/' in f])

    if len(subdirectory)>0:
        if any('.csv' in filename for filename in os.listdir(data_root)):
            df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
            data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    mt_imgfiles, _, _, _, _ = amp.io.tiff_walk(data_root, parse_channels=True)

    df_info = amp.io.parse_filename(data_root)

    num_pb = df_info['photobleach frame number'].values[0]

    if num_pb > len(mt_imgfiles):
        return
    elif any(filename in data_root for filename in zerolist):
        num_pb = 0

    if ('DLP_Red_trimmed'.encode('ASCII') not in os.listdir(data_root)) and ('DLP_Red_trimmed' not in os.listdir(data_root)):
        os.mkdir(os.path.join(data_root,'DLP_Red_trimmed'))

    if 'mt647_mt488' in data_root:
        if 'DLP_Blue_trimmed' not in os.listdir(data_root):
            os.mkdir(os.path.join(data_root,'DLP_Blue_trimmed'))

    # Find center and size of network in frame before photobleaching
    y_camcent = 650
    x_camcent = 1000
    wincam = 500
    camera_crop = create_window(x_camcent, y_camcent, wincam)

    df_info = amp.io.parse_filename(data_root)
    
    # Find center of network using image immediately before photobleaching occurs
    if (num_pb > 0) or ('11-17-2022_slide1_lane2_pos4_k401bac_iLidmicro_longMT647_5s_intervals_200ms_DLPRed_50ms_DLPBlue_skip0_photobleach_frame2_1' in data_root):
        im = io.imread(mt_imgfiles[num_pb - 1])[camera_crop]
        x_cent, y_cent, winsize = initial_center(im, show_process=show_process)
    else:
        print(os.path.split(data_root)[-1], ' requires crude center identification')
        x_cent, y_cent, winsize = 250, 250, 500

    for n in range(num_pb, num_pb + n_tot):

        if n == len(mt_imgfiles):
            break
        im = io.imread(mt_imgfiles[n],as_gray=True)
        im_camcropped = im[camera_crop]

        if 'mt647_mt488' in data_root:
            im_blue = io.imread(blue_imgfiles[n], as_gray=True)
            im_bluecropped = im_blue[camera_crop]

        if winsize>430:
            io.imsave(os.path.join(data_root,'DLP_Red_trimmed','DLP_Red_%03d.tif' %(n-num_pb)), im_camcropped, 
                      check_contrast=False)
            if 'mt647_mt488' in data_root:
                io.imsave(os.path.join(data_root, 'DLP_Blue_trimmed', 'DLP_Blue_%03d.tif' %(n-num_pb)), im_bluecropped, 
                        check_contrast=False)

        else:
            try:
                im_cropped, x_cent, y_cent = crop_network(im_camcropped, x_cent, y_cent, winsize, show_process=show_process)
                
            except:
                new_win = create_window(y_cent, x_cent, winsize)
                im_cropped = im_camcropped[new_win]

            io.imsave(os.path.join(data_root,'DLP_Red_trimmed','DLP_Red_%03d.tif' %(n-num_pb)),im_cropped, 
                      check_contrast=False)
            if 'mt647_mt488' in data_root:
                new_win = create_window(y_cent, x_cent, winsize)
                io.imsave(os.path.join(data_root,'DLP_Blue_trimmed', 'DLP_Blue_%03d.tif' %(n-num_pb)), im_blue[camera_crop][new_win], 
                        check_contrast=False)
    
    return

def simple_crop(data_root, n_tot=20, show_process=False):
    """
    Performs a simple cropping of n_tot images from the image stack starting from num_pb

    Inputs :
    ----------
    data_root : str, root directory of image set
    
    n_tot : int, default 20, number of frames after photobleaching to crop
    show_process : bool, default False, provides plots of cropped images

    Returns :
    -----------
    files : saved in data_root under their 'DLP_Red_trimmed' and for two-color experiments, with 'DLP_Blue_trimmed'
    """
    zerolist = ['210426_slide1_lane1_pos1', '210426_slide1_lane1_pos2', '210426_slide1_lane2_pos1']
    _, _, _, _, subdirectory = amp.io.tiff_walk(data_root, parse_channels=True)

    if 'mt647_mt488' in data_root:
        files = amp.io.find_all_tiffs(data_root)
        blue_imgfiles = np.sort([f for f in files if '/DLPBlue/' in f])

    if len(subdirectory)>0:
        if any('.csv' in filename for filename in os.listdir(data_root)):
            df_csv = pd.read_csv(os.path.join(data_root, os.path.split(data_root)[-1]+ '_filename_order.csv'), sep=',')
            data_root = df_csv[df_csv['order']==1]['filename'].values[0]
    mt_imgfiles, _, _, _, _ = amp.io.tiff_walk(data_root, parse_channels=True)

    df_info = amp.io.parse_filename(data_root)

    num_pb = df_info['photobleach frame number'].values[0]

    if num_pb > len(mt_imgfiles):
        return
    elif any(filename in data_root for filename in zerolist):
        num_pb = 0

    if ('DLP_Red_trimmed'.encode('ASCII') not in os.listdir(data_root)) and ('DLP_Red_trimmed' not in os.listdir(data_root)):
        os.mkdir(os.path.join(data_root,'DLP_Red_trimmed'))

    # Find center and size of network in frame before photobleaching
    y_camcent = 600
    x_camcent = 1000
    wincam = 500
    camera_crop = create_window(x_camcent, y_camcent, wincam)

    df_info = amp.io.parse_filename(data_root)

    for n in range(num_pb, num_pb + n_tot):
        
        if n == len(mt_imgfiles):
            break
        im = io.imread(mt_imgfiles[n],as_gray=True)
        im_camcropped = im[camera_crop]

        io.imsave(os.path.join(data_root,'DLP_Red_trimmed','DLP_Red_%03d.tif' %(n-num_pb)), im_camcropped, 
                    check_contrast=False)

        if 'mt647_mt488' in data_root:
            im_blue = io.imread(blue_imgfiles[n], as_gray=True)
            im_bluecropped = im_blue[camera_crop]
            io.imsave(os.path.join(data_root,'DLP_Blue_trimmed', 'DLP_Blue_%03d.tif' %(n-num_pb)), im_blue[camera_crop], 
                        check_contrast=False)

    return