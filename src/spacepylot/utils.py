# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:50:06 2021

@author: Liz_J
"""
__author__ = "Elizabeth Watkins, Eric Emsellem"
__copyright__ = "Elizabeth Watkins"
__license__ = "MIT License"
__contact__ = "<liz@email"

# General imports from numpy and copy
import numpy as np
import copy as cp
from functools import lru_cache, wraps

# ODR import
from scipy.odr import ODR, Model, RealData
import scipy.ndimage as ndi

# Skimage
from skimage import exposure

# robust stats
from astropy.stats import mad_std, sigma_clip

# Internal package calls
from .params import pcc_params
from . import alignment_utilities as au

default_kw_opticalflow = {'attachment': 10, 'tightness': 0.3, 'num_warp': 15,
                          'num_iter': 10, 'tol': 0.0001, 'prefilter': False}


class VerbosePrints:
    """
    General print statements to inform the user what the alignment algorithms
    have found. Only prints if the user asks for verbose mode to be switched
    on
    """

    def __init__(self, verbose):
        """
        Initialised the print statements

        Parameters
        ----------
        verbose : bool
            Tells the class whether to print or not.

        Returns
        -------
        None.

        """
        self.verbose = verbose
        if self.verbose:
            print('Verbose mode on')
        # No need to say "verbose is off" if it is off.
        # else:
        #     print("Verbose mode off")

    def default_filter_params(self):
        """
        Prints the default filter parameters if no parameters
        are set by the user to inform them how the images are filtered
        before the analysis is run on them

        Returns
        -------
        None.

        """
        if self.verbose:
            print('\nRemoving boundary of %d. Filtering image using '
                  'histogram equalisation and high pass filter %s with arguments:'
                  % (pcc_params['remove_boundary_pixels'], pcc_params['hpf'].__name__))
            for key in pcc_params['hpf_kwargs'].keys():
                print(key, pcc_params['hpf_kwargs'][key])

    def applied_translation(self, yx_offset):
        """
        Prints whenever the user manually applies a translational offset
        to the prealign image

        Parameters
        ----------
        yx_offset : 2-array
            The y and x offset applied to the prealign image.
            Positive values shift the prealign in the negative direction i.e
            A coordinate of 3,3, with an offset of 1,1 will be shifted to 2,2.

        Returns
        -------
        None.

        """
        if self.verbose and np.sum(yx_offset) != 0:
            print(f"\nAdditional offset of:"
                  f"\ny={yx_offset[0]:.4f}, x={yx_offset[1]:.4f} pixels has been applied.")

    def applied_rotation(self, rotation_angle):
        """
        Prints whenever the user manually applies a translational offset
        to the prealign image

        Parameters
        ----------
        rotation_angle : float
            Angle, in degrees, of the rotational offset applied to the prealign
            image

        Returns
        -------
        None.

        """

        if self.verbose and rotation_angle != 0:
            print('\nAdditional rotation of: \n'
                  f'theta={rotation_angle:4f} degrees '
                  'has been applied.')

    def get_translation(self, shifts, added_shifts=(0, 0)):
        """
        Prints the translation found by the align object

        Parameters
        ----------
        shifts : 2-array
            The y and x offset needed to align the prealign image with the
            reference image.  Positive values shift the prealign in the negative direction i.e
            A coordinate of 3,3, with an offset of 1,1 will be shifted to 2,2.
        added_shifts : 2-array, optional
            Additional shifts already applied to the image. The default is [0,0].

        Returns
        -------
        None.

        """
        if self.verbose:
            print('\n--------Found translational offset---------')
            print('Offset of %g y, %g x pixels found' % tuple(shifts))

            if np.sum(added_shifts) != 0:
                shifty = shifts[0] + added_shifts[0]
                shiftx = shifts[1] + added_shifts[1]
                print('Total offset of %g y, %g x pixels found' % (shifty, shiftx))

    def get_rotation(self, rotation, added_rotation=0):
        """
        Prints the rotational offset found by the align object

        Parameters
        ----------
        rotation : float
            Angle, in degrees,  needed to align the prealign image with the
            reference image
        added_rotation : float, optional
            Additional rotation already applied to the image. The default is 0.

        Returns
        -------
        None.

        """
        if self.verbose:
            print('\n----------Found rotational offset----------')
            print('Rotation of %.4f degrees found' % rotation)
            if added_rotation != 0:
                tot_rot = rotation + added_rotation
                print('Total rotation of %.4f degrees found' % tot_rot)


def _split(size, num):
    """
    For a given length, and division position within an axis, works
    out the start and end indices of that sub-section of the axis.
    If an image has been subdivided 2 times (4 quadrants), so
    the length of the quadrant sides are axis/2, `num`=0 is used
    to get the start and end indices of that quadrant.

    Parameters
    ----------
    size : int
        Length of subaxis.
    num : int
        start location of subaxis.

    Returns
    -------
    lower : int
        Lower index position of a given length.
    upper : int
        Upper index position of a given length.

    """
    lower, upper = np.int(size) * np.array([num, num+1])
    return lower, upper


def reject_outliers(data, m=3):
    """Rejects outliers given by a factor of the standard deviation

    Parameters
    ----------
    data: numpy array
        Input 1D array to filter
    m : float
        Input factor which gives the number of standard deviations
        beyond which points will be rejected.
    """
    return data[np.abs(data - np.nanmedian(data)) < m * np.nanstd(data)]


def _remove_nonvalid_numbers(image):
    """
    Removes non valid numbers from the array and replaces them with zeros

    Parameters
    ----------
    image : 2darray
        Image that will have its non valid values replaced with zeros

    Returns
    -------
    image_valid : 2darray
        Image without any NaNs or infs

    """
    return cp.deepcopy(np.nan_to_num(image))


def _remove_image_border(image, border):
    """
    Shrinks the image by removing the given border value long each edge
    of the image

    Parameters
    ----------
    image : 2darray
        Image that will have its borders removed
    border : int
        Number of pixels to remove from each edge.


    Returns
    -------
    image_border_removed : 2darray
        Image with boundary pixels removed Returned in a
        nd-2*border, nd-2*border array.

    """
    # data_shape = min(np.shape(image))
    # image = cp.deepcopy(image[:data_shape, :data_shape])
    image_border_removed = image[border:-border, border:-border]

    return image_border_removed


def chunk_stats(list_arrays, chunk_size=15):
    """Cut the datasets in 2d chunks and take the median
    Return the set of medians for all chunks.

    Parameters
    ----------
    list_arrays : list of np.arrays
        List of arrays with the same sizes/shapes
    chunk_size : int
        number of pixel (one D of a 2D chunk)
        of the chunk to consider (Default value = 15)

    Returns
    -------
    median, standard: 2 arrays of the medians and standard deviations
        for the given datasets analysed in chunks.

    """
    # Check that all arrays have the same size
    if not np.all([list_arrays[0].size == d.size for d in list_arrays[1:]]):
        #Relic from pymusepipe. Changing to raise error
        # upipe.print_error("Datasets are not of the same "
        #                   "size in median_compare")
        raise IndexError("Arrays given in `list_arrays` are not of the same"\
                          "size in `chunk_stats`")

    list_arrays = np.atleast_3d(list_arrays)
    narrays = len(list_arrays)

    nchunk_x = np.int32(list_arrays[0].shape[0] // chunk_size) #-1)
    nchunk_y = np.int32(list_arrays[0].shape[1] // chunk_size) #-1)

    grid_number = nchunk_x * nchunk_y

    #Vectorised to remove 3-nest loop
    arrays3d_chunk_ready = list_arrays[:,:nchunk_x*chunk_size,:nchunk_y*chunk_size]

    grids_nchunkx  = np.array(np.split(arrays3d_chunk_ready, nchunk_x, axis=1))

    grids_xy = np.array(np.split(np.array(grids_nchunkx), nchunk_y, axis=-1))

    med_array = np.nanmedian(grids_xy, axis=(-2,-1)).T.reshape(narrays, grid_number)
    std_array = mad_std(grids_xy, axis=(-2,-1), ignore_nan=True).T.reshape(narrays, grid_number)

    # Cleaning in case of Nan
    med_array = np.nan_to_num(med_array)
    std_array = np.nan_to_num(std_array)
    return med_array, std_array

def get_polynorm_SP(array1, array2, chunk_size=15, threshold1=0.,
                        threshold2=0, percentiles=(0., 100.), sigclip=0):
    """Find the normalisation factor between two arrays.
    Including the background and slope. This uses the function
    regress_odr which is included in align_pipe.py and itself
    makes use of ODR in scipy.odr.ODR.
    Parameters
    ----------
    array1 : 2D np.array
    array2 : 2D np.array
        2 arrays (2D) of identical shapes
    chunk_size : int
        Default value = 15
    threshold1 : float
        Lower threshold for array1 (Default value = 0.)
    threshold2 : float
        Lower threshold for array2 (Default value = 0)
    percentiles : list of 2 floats
        Percentiles (Default value = [0., 100.])
    sigclip : float
        Sigma clipping factor (Default value = 0)
    Returns
    -------
    result: python structure
        Result of the regression (ODR)
    """

    # proceeds by splitting the data arrays in chunks of chunk_size
    med, std = chunk_stats([array1, array2], chunk_size=chunk_size)

    # Selecting where data is supposed to be good
    if threshold1 is None:
        threshold1 = 0.
    if threshold2 is None:
        threshold2 = 0.
    pos = (med[0] > threshold1) & (std[0] > 0.) & (std[1] > 0.) & (med[1] > threshold2)

    # Guess the slope from this selection
    guess_slope = 1.0

    # Doing the regression itself
    result = regress_odr(x=med[0][pos], y=med[1][pos], sx=std[0][pos],
                         sy=std[1][pos], beta0=[0., guess_slope],
                         percentiles=percentiles, sigclip=sigclip)
    result.med = med
    result.std = std
    result.selection = pos
    return result

def my_linear_model(B, x):
    """Linear function for the regression.

    Parameters
    ----------
    B : 1D np.array of 2 floats
        Input 1D polynomial parameters (0=constant, 1=slope)
    x : np.array
        Array which will be multiplied by the polynomial

    Returns
    -------
        An array = B[1] * (x + B[0])
    """
    return B[1] * (x + B[0])


def regress_odr(x, y, sx, sy, beta0=(0., 1.),
                percentiles=(0., 100.), sigclip=0.0):
    """Return an ODR linear regression using scipy.odr.ODR

    Parameters
    ----------
    x : numpy.array
    y : numpy.array
        Input array with signal
    sx : numpy.array
    sy : numpy.array
        Input array (as x,y) with standard deviations
    beta0 : list or tuple of 2 floats
        Initial guess for the constant and slope
    percentiles: tuple or list of 2 floats
        Two numbers providing the min and max percentiles
    sigclip: float
        sigma factor for sigma clipping. If 0, no sigma clipping
        is performed

    Returns
    -------
    result: result of the ODR analysis

    """
    # Percentiles
    xrav = x.ravel()
    if len(xrav) > 0:
        percentiles = np.nanpercentile(xrav, percentiles)
        sel = (xrav >= percentiles[0]) & (xrav <= percentiles[1])
    else:
        sel = np.abs(xrav) > 0

    xsel, ysel = xrav[sel], y.ravel()[sel]
    sxsel, sysel = sx.ravel()[sel], sy.ravel()[sel]
    linear = Model(my_linear_model)

    # We introduce the minimum of x to avoid negative values
    minx = np.min(xsel)
    mydata = RealData(xsel - minx, ysel, sx=sxsel, sy=sysel)
    result = ODR(mydata, linear, beta0=beta0)

    if sigclip > 0:
        diff = ysel - my_linear_model([result.beta[0], result.beta[1]], xsel)
        filtered = sigma_clip(diff, sigma=sigclip)
        xnsel, ynsel = xsel[~filtered.mask], ysel[~filtered.mask]
        sxnsel, synsel = sxsel[~filtered.mask], sysel[~filtered.mask]
        clipdata = RealData(xnsel, ynsel, sx=sxnsel, sy=synsel)
        result = ODR(clipdata, linear, beta0=beta0)

    # Running the ODR
    r = result.run()
    # Offset from the min of x
    r.beta[0] -= minx

    return r


def filtermed_image(data, border=0, filter_size=2):
    """Process image by removing the borders
    and filtering it via a median filter

    Input
    -----
    data: 2d array
        Array to be processed
    border: int
        Number of pixels to remove at each edge
    filter_size: float
        Size of the filtering (median)

    Returns
    -------
    cdata: 2d array
        Processed array
    """
    # Omit the border pixels
    if border > 0:
        data = crop_data(data, border=border)

    return ndi.filters.median_filter(data, filter_size)


def filter_image_for_analysis(image, histogram_equalisation=False,
                              remove_boundary_pixels=25, convolve=None,
                              hpf=None, hpf_kwargs=None):
    """
    The function that controls how the prealign and reference image are
    filtered before running the alignment

    Parameters
    ----------
    image : 2d array
        Image to apply the filters to.
    histogram_equalisation : Bool, optional
        If true, scales the intensity values according to a histogram
        equalisation. This tends to help computer vision find features
        as it maximises the contrast. The default is False.
    remove_boundary_pixels : int, optional
        Removes pixels around the image if there are bad pixels at the
        detector edge. The default is 25.
    convolve : int or None, optional
        If a number, it will convolve the image. The number refers to
        the sigma of the folding Gaussian to convolve by in units of pixels
        (I think). The default is None
    hpf : function, optional
        The high pass filter function to use to filter out high frequencies.
        Higher frequencies can reduce the performance of alignment
        routines. The default is None.
    hpf_kwargs : dict, optional
        The dictionary arguments needed for `hpf`. The default is {}.

    Returns
    -------
    image : 2d array
        The filtered image.

    """

    image = _remove_image_border(image, remove_boundary_pixels)
    image = _remove_nonvalid_numbers(image)
    if convolve is not None:
        image = au.convolve_image(image, convolve)

    # This helps to match up the intensities
    if histogram_equalisation:
        image = exposure.equalize_hist(image)

    # Remove some frequencies to make alignment more robust vs noise
    if hpf is not None:
        if hpf_kwargs is None:
            hpf_kwargs = {}
        image = hpf(image, **hpf_kwargs)

    return image

if __name__ == "__main__":
    pass