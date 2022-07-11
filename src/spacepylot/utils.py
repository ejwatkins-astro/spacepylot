# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:50:06 2021

@author: Liz_J
"""
__author__ = "Elizabeth Watkins, Eric Emsellem"
__copyright__ = "Elizabeth Watkins"
__license__ = "MIT License"
__contact__ = "<liz@email"

import numpy as np
import copy as cp

from skimage import exposure

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
        Initalised the print statments

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
        else:
            print('Verbose mode off')

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
    lower, upper = int(size) * np.array([num, num+1])
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
    return data[abs(data - np.median(data)) < m * np.std(data)]


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
