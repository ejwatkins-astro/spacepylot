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
from scipy import ndimage as ndi

# more standard modules
import copy as cp

# Utilities from skimage
from skimage import transform
from skimage.transform import rotate
from skimage.measure import ransac

# Reproject
from reproject import reproject_interp

# Astropy
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from astropy import wcs


def column_stack_2d(array_2d_first, array_2d_second):
    """
    Converts equal dimension 2d arrays into index matched values

    Parameters
    ----------
    array_2d_first : 2d array
        Th array to fill column 1.
    array_2d_second : 2d array
        Th array to fill column 2.

    Returns
    -------
    array_stacked : n*mx2 array
        Contains the column stacked values of the given arrays.

    """

    array_2d_first_flat = array_2d_first.flatten()
    array_2d_second_flat = array_2d_second.flatten()

    array_stacked = np.column_stack([array_2d_first_flat, array_2d_second_flat])

    return array_stacked


def create_euclid_homography_matrix(rotation, translation, rotation_first=True):
    """
    Creates the homography representation of a given rotation and translation.

    Parameters
    ----------
    rotation : float
        Angle, in degrees, of the rotational offset
    translation : 2-array
        An array containing 2 numbers. First number is the y shift to
        be applied, second is the x
    rotation_first : bool, optional
        Homography perform rotation first, the translation. To reverse this,
        set bool to False. The default is True.

    Returns
    -------
    homography : 3x3 array
        The 3x3 homography matrix representing the rotation and translation
        of an image

    """
    y, x = translation
    rotation_radians = np.deg2rad(rotation)
    cos_theta, sin_theta = np.cos(rotation_radians), np.sin(rotation_radians)

    if rotation_first:
        a, b = x, y
    else:
        a = x * cos_theta + y * sin_theta
        b = y * cos_theta - x * sin_theta

    homography = np.array([
        [cos_theta, -sin_theta, a],
        [sin_theta, cos_theta, b],
        [0, 0, 1]
        ]
    )
    return homography


def homography_on_grid_points(original_xy, transformed_xy,
                              method=transform.EuclideanTransform,
                              reverse_order=False,
                              **kwargs):
    """
    Finds the offset between two images by estimating the matrix transform needed
    to reproduce the vector changes between a set of xy coordinates.

    This can align an image given a minimum of 4 xy, grid points from the reference
    image and their location in the prealign image.

    Default homography is the skimage.transform.EuclideanTransform. This forces
    the solution to output only a rotation and a translation.
    Other homography matrices can be used to find image shear, and scale changed
    between th reference and prealign image.

    Homography works because the translation and rotation (and scale etc.) matrix
    can be combined into one operation. This matrix is called the homography
    matrix. We can then write out a set of equations
    describing the new yx, grid points as a function of the rotation and translation.
    By writing these equations for the new x and y for each grid point
    up as a matrix, we can just use singular value decomposition (SVD) to find the
    coefficients (ie the translation and rotation) and out.
    To improve this, we typically use least squares, with additional constraints
    that we now to help improve the output homography matrix.

    Most off the shell homography routines apply the rotation matrix first
    then apply the translation matrix. To reverse this ordering, set
    `self.reverse_order` in the method `homography_on_grid_points`
    to True

    This function performs a fit to estimate the homography matrix that 
    represents the grid transformation. Uses ransac 
    to robustly eliminate vector outliers that
    something such as optical flow, or other feature extracting methods
    might output. The parameters are not currently changeable by the user
    since not entirely sure what are the best parameters to use and so
    are using values given in skimage tutorials.
    for Euclidean transformations (i.e., pure rotations and translations)

    Parameters
    ----------
    original_xy : nx2 array
        nx2 array of the xy coordinates of the reference grid.
    transformed_xy :  nx2 array
        nx2 array of the xy coordinates of a feature that has been transformed
        by a rotation and translation (coordinates of object in prealign grid)
    method : function, optional
        method is the what function is used to find the homography.
        The default is transform.EuclideanTransform.
    reverse_order : bool, optional
        Homography matrix outputs the solution representing a rotation followed
        by a translation. If you want the solution to translate, then rotate
        set this to True. The default is False.

    Returns
    -------
    shifts : 2-array
        The recovered shifts, in pixels.
    rotation : float
        The recovered rotation, in degrees.
    homography_matrix : 3x3 matrix
        The Matrix that has been "performed" that resulted in the offset
        between the prealign and the reference. The inverse therefore
        describes the parameters needed to convert the prealign image
        to the reference grid.

    """
    # Get keywords from kwargs
    # If test value is True then keeping this empty to use default from ransac
    # and speed up things
    kwargs_ransac = {'min_samples': 3, 'residual_threshold': 0.5,
                     'max_trials': 1000}

    # Overwriting the keywords in case those are provided
    for key in kwargs:
        kwargs_ransac[key] = kwargs.pop(key)

    # Ransac call
    model_robust, inliers = ransac((original_xy, transformed_xy), method,
                                   **kwargs_ransac)

    # If you want the matrix to represent translation then rotation, set this
    # to True
    homography_matrix = model_robust.params
    shifts = get_shifts_from_homography_matrix(homography_matrix,
                                               reverse_order)
    # rotation = np.rad2deg(np.arcsin(model_robust.rotation))
    rotation = get_rotation_from_homography_matrix(homography_matrix)

    return shifts, rotation, homography_matrix


def get_shifts_from_homography_matrix(homography_matrix, 
                                      reverse_order=False):
    """Extracting the shift from an homography matrix

    Parameters
    ----------
    homography_matrix: 3x3 matrix
        Homography matrix
    reverse_order: bool
        If True we need to use correct_homography_order to get
        the shifts. Default to False

    Returns
    -------
    shifts: [float, float]
     
    """
    if homography_matrix is None:
        return [0., 0.]

    if reverse_order:
        shifts = correct_homography_order(homography_matrix)
    else:
        shifts = cp.copy(homography_matrix[:2, -1][::-1])

    return shifts


def get_rotation_from_homography_matrix(homography_matrix):
    """Extracting the rotation from an homography matrix

    Parameters
    ----------
    homography_matrix: 3x3 matrix
        Homography matrix

    Returns
    -------
    rotation: float [in degrees]
     
    """
    if homography_matrix is None:
        return 0.
    else:
        return np.arctan2(homography_matrix[1, 0], 
                          homography_matrix[1, 1])


def correct_homography_order(homography_matrix):
    """
    Homography order is rotation then translation. Since rotation and then
    translation are non-commutative, the homography matrix would need
    to change if translation, then rotation is applied. If translation
    then rotation was applied, use this method to adjust off-the-shelf
    homography finding method to match this order of transformation.
    This only works for Euclidean transforms (i.e., those limited to
    rotation and translation) If affine/projective etc. transforms have been
    used, this will not be the correct solution

    Parameters
    ----------
    homography_matrix : 3x3 matrix
        Euclidean transform homography matrix containing the rotational
        and translational information.

    Returns
    -------
    2-array
        Returns the adjusted shifts for translation followed by rotation.

    """

    a = homography_matrix[0, -1]
    b = homography_matrix[1, -1]
    cos_theta = homography_matrix[0, 0]
    sin_theta = homography_matrix[1, 0]

    dy = b * cos_theta - a * sin_theta
    dx = a * cos_theta + b * sin_theta

    return np.array([dy, dx])


def translate_image(image, shifts):
    """
    Applies simple translation shift while preserving the input dimensions

    Parameters
    ----------
    image : 2darray
        Image to apply translational offset to.
    shifts : y,x array of 2 float
        An array containing 2 numbers. First number is the y shift to
        be applied, second is the x

    Returns
    -------
    image_translated : 2darray
        Image with new coordinate grid translated according to given
        `shifts`. Input dimensions preserved

    """
    image_no_nans = np.nan_to_num(image)
    image_translated = cp.deepcopy(ndi.shift(image_no_nans, shifts))

    return image_translated


def rotate_image(image, angle_degrees):
    """
    Applies simple rotational shift while preserving the input dimensions

    Parameters
    ----------
    image : 2darray
        Image to apply rotational offset to.
    angle_degrees : float
        Angle, in degrees, of the rotational offset to be applied to `image`.

    Returns
    -------
    image_rotated : 2darray
        Image with new coordinate grid rotated according to given
        `angle_degrees`. Input dimensions preserved.

    """
    image_no_nans = np.nan_to_num(image)
    image_rotated = cp.deepcopy(rotate(image_no_nans, angle_degrees))

    return image_rotated


def transform_image_wcs(image, header, rotation=0, yx_offset=None):
    """Rotates and translate image using reproject keeping the 
    image size the same

    Parameters
    ----------
    image : 2d image
        image to warp.
    header: astropy.io.Header.header
        Dictionary-like object containing the world coordinate reference.
    rotation : float, optional
        Rotation angle to apply to the image in degrees. The default is 0.
    yx_offset : 2-array, optional
        The yx coordinate to translate the image. The default is [0,0].

    Returns
    -------
    transformed_image : 2d array
        image that has been warped to new grid.
    """
    # Getting the WCS
    input_wcs = wcs.WCS(naxis=2)
    input_wcs.wcs.crpix = header['CRPIX1'], header['CRPIX2']

    try: # TODO check if this is consistent if CD1_2 and  CD2_1 are given etc
        input_wcs.wcs.cdelt = header['CD1_1'], header['CD2_2']
    except KeyError:
        input_wcs.wcs.cdelt = header['CDELT1'], header['CDELT2']

    if yx_offset is None:
        yx_offset = np.zeros(2)
    output_wcs = wcs.WCS(naxis=2)
    output_wcs.wcs.crpix = input_wcs.wcs.crpix + yx_offset[::-1]
    output_wcs.wcs.cdelt = input_wcs.wcs.cdelt
    rot = -np.deg2rad(rotation)
    output_wcs.wcs.pc = [[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]]

    transformed_image, _ = reproject_interp((image, input_wcs),
                                            output_wcs,
                                            shape_out=[header['NAXIS2'],
                                                       header['NAXIS1']])
    return transformed_image


def convolve_image(image, sigma):
    """
    Convolve the image with the sigma supplied. Sigma I think, related
    to the pixel size of the image

    Parameters
    ----------
    image : 2d-array
        Image to be convolved.
    sigma : float
        Standard deviation of the Gaussian used to convolve the image with.
        I think the units used by the function are in number of image pixels

    Returns
    -------
    image : 2d-array
        Convolved image.

    """
    if sigma > 0:
        kernel = Gaussian2DKernel(x_stddev=sigma)
        image = cp.deepcopy(convolve(image, kernel))

    return image


def sparse_2d_grid(array_2d, num_per_dimension=20, return_step=False):
    """
    From a given 2d array, generates a grid of y, x indices with steps
    determined by the number of wanted grid points per dimension.

    Parameters
    ----------
    array_2d : 2darray
        2d data array to find y, x grid.
    num_per_dimension : int, optional
        For each axis, this sets the number of grid points to be calculated.
        The default is 20
    return_step : bool, optional
        The step size between grid points needed to display the given
        `num_per_dimension` amount of grid points. The default is False.

    Returns
    -------
    y_inds_sparse : 2darray
        grid points int the y direction sampling the given 2darray according
        to the number of grid points requested with parameter `num_per_dimension`
    x_inds_sparse : 2darray
        grid points int the x direction sampling the given 2darray according
        to the number of grid points requested with parameter `num_per_dimension`
    step : int, optional
        Optional parameter to be returned if `return_step` is set to True.
        Returned is the step size between grid points used when initialising
        the grid

    """
    rows, cols = array_2d.shape
    smallest = min(rows, cols)
    if num_per_dimension > smallest:
        num_per_dimension = smallest

    step = max(rows//num_per_dimension, cols//num_per_dimension)

    y_inds_sparse, x_inds_sparse = np.mgrid[:rows:step, :cols:step]
    if return_step:
        return y_inds_sparse, x_inds_sparse, step
    else:
        return y_inds_sparse, x_inds_sparse


def get_sparse_vector_grid(v, u, num_per_dimension=20):
    """
    Gets a sparse grid for vector components along with the sparse yx
    positions of the grid. This is needed when calculating homography from
    the vector grid to speed up the operation and needed when
    visualising/displaying the vector field

    Parameters
    ----------
    v : 2darray
        Vector component in the y direction.
    u : 2darray
        Vector component in the x direction.
    num_per_dimension : int, optional
        For each axis, this sets the number of grid points to be calculated.
        The default is 20

    Returns
    -------
    y_sparse : 2darray
        2d array of shape `num_per_dimension`, `num_per_dimension`. containing
        the y coordinates sampled from the vector grid.
    x_sparse : 2darray
        2d array of shape `num_per_dimension`, `num_per_dimension`. containing
        the x coordinates sampled from the vector grid.
    v_sparse : 2darray
        2d array of shape `num_per_dimension`, `num_per_dimension`. containing
        the y component vectors sampled from the y components of the vector grid.
    u_sparse : 2darray
        2d array of shape `num_per_dimension`, `num_per_dimension`. containing
        the x component vectors sampled from the y components of the vector grid.

    """
    y_sparse, x_sparse, step = sparse_2d_grid(
        array_2d=u,
        num_per_dimension=num_per_dimension,
        return_step=True
    )

    u_sparse = u[::step, ::step]
    v_sparse = v[::step, ::step]

    return y_sparse, x_sparse, v_sparse, u_sparse


def open_fits(filepath, hdu_i=0):
    with fits.open(filepath) as hdulist:
        data = hdulist[hdu_i].data
        header = hdulist[hdu_i].header
    return data, header


def _umeyama_translation(reference_coords, prealign_coords):
    """Estimates the similarity transformation for translational
    offsets only.
    Parameters
    ----------
    reference_coords : (M, N) array
        Reference coordinates.
    prealign_coords : (M, N) array
        Prealign coordinates.
    Returns
    -------
    homography_solution : (N + 1, N + 1)
        The homogeneous similarity transformation matrix.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    .. [2] Code modified from skimage.transform._geometric
    """
    dimension = reference_coords.shape[1]

    # Compute mean x and y coordinates for the reference and prealign.
    reference_coords_mean = np.nanmean(reference_coords, axis=0)
    prealign_coords_mean = np.nanmean(prealign_coords, axis=0)
    homography_solution = np.identity(dimension + 1, dtype=np.double)

    # [1] eq 41. : translation = prealign_coords_mean - scale * (rotation_matrix @ reference_coords_mean.T)
    # translation, without scale and rotation, has the following solution:
    homography_solution[:dimension, dimension] = prealign_coords_mean - reference_coords_mean

    return homography_solution


class TranslationTransform(transform.ProjectiveTransform):
    """Translation transformation
    Has the following form::
        X = a0 * x - b0 * y + a1 =
          = x * cos(0) - y * sin(0) + a1
          = x + a1
        Y = b0 * x + a0 * y + b1 =
          = x * sin(0) + y * cos(0) + b1
          = y + b1
    where the homogeneous transformation matrix is::
        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]
        =
        [[1  0  a1]
         [0  1  b1]
         [0  0   1]]
    This transformation is a rigid transformation with translation parameters only.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix.
    translation : sequence of float, length D, optional
        Translation parameters for each axis.
    dimensionality : int, optional
        The dimensionality of the transform.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    .. [2]  Code modified from skimage.transform.EuclideanTransform
    """

    def __init__(self, matrix=None, translation=None,
                 *, dimensionality=2):
        params_given = translation is not None

        if params_given and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix

        elif params_given:

            if translation is None:
                translation = (0,) * dimensionality

            if dimensionality == 2:
                self.params = np.identity(3)

            elif dimensionality == 3:
                self.params = np.eye(dimensionality + 1)

            self.params[0:dimensionality, dimensionality] = translation

        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.
        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.
        Number of source and destination coordinates must match.
        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        References
        ----------
        .. [2]  Code modified from skimage.transform.EuclideanTransform
        """
        # umeyama is the author of the person who invented this method
        self.params = _umeyama_translation(src, dst)

        # _umeyama will return nan if the problem is not well-conditioned.
        return not np.any(np.isnan(self.params))

    # kept a rotation output of zero to help avoid errors
    @property
    def rotation(self):
        return 0

    @property
    def translation(self):
        return self.params[0:2, 2]
