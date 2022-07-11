# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:44:01 2021
Updated 2021-2022 - EE

@author: Liz_J
"""
__author__ = "Elizabeth Watkins, Eric Emsellem"
__copyright__ = "Elizabeth Watkins"
__license__ = "MIT License"
__contact__ = "<liz@email>"


# Importing modules
import numpy as np

# Skimage
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from skimage import transform
from skimage.measure import ransac

# copy
import copy as cp

# Internal calls
from . import alignment_utilities as au
from .alignment_utilities import (get_shifts_from_homography_matrix,
                                  get_rotation_from_homography_matrix)
from .utils import VerbosePrints, filter_image_for_analysis, _split, default_kw_opticalflow
from .params import pcc_params


class HomoMatrix(object):
    """
    Class to embed a 3x3 homographic or transformation matrix
    """
    def __init__(self, homo_matrix=None, reverse_order=False):
        """Initialise the homography matrix and the reverse options

        Parameters
        ----------
        homo_matrix : 3x3 array, optional
            3x3 array containing the homograpic solution. The default is None.
        reverse_order : bool, optional
            Updates the translation solution so that is is correct for offset
            updates that start with translation then rotation. The default is
            False.

        Returns
        -------
        None.

        """
        if homo_matrix is None:
            self.homo_matrix = np.identity(3)
        else:
            self.homo_matrix = homo_matrix
        self.reverse_order = reverse_order

    def __repr__(self):
          return 'HomoMatrix(\n%s)' % str(self.homo_matrix)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self,index):
        return self.homo_matrix[index]

    @property
    def translation(self):
        """Returns the translation shifts from the homography
        matrix
        """
        return get_shifts_from_homography_matrix(self.homo_matrix,
                                                 self.reverse_order)

    @property
    def rotation_rad(self):
        """Returns the rotation in radians from the homography
        matrix
        """
        return get_rotation_from_homography_matrix(self.homo_matrix)

    @property
    def rotation_deg(self):
        """Returns the rotation in degrees from the homography
        matrix
        """
        return np.rad2deg(self.rotation_rad)

    def __matmul__(self, b):
        """Returns a new HomoMatrix using the multiplication
        with self as first operand (B being the second).
        Important note: we always copy the first HomoMatrix
        to keep the reverse order and translation options

        Parameters
        ----------
        b : 3x3 array or HomoMatrix

        Returns
        -------
        HomoMatrix with the self @ B multiplication

        """
        newhm = cp.copy(self)
        if isinstance(b, HomoMatrix):
            b = b.homo_matrix
        return HomoMatrix(self.homo_matrix @ b, self.reverse_order)


    def __rmatmul__(self, a):
        """Returns a new HomoMatrix using the multiplication
        with self as the second operand (A being the first).
        Important note: we always copy the first HomoMatrix
        to keep the reverse order and translation options

        Parameters
        ----------
        a : 3x3 array or HomoMatrix

        Returns
        -------
        HomoMatrix with the A @ self multiplication
        """

        if isinstance(a, HomoMatrix):
            a = a.homo_matrix
        return HomoMatrix(a @ self.homo_matrix, self.reverse_order)

    __array_priority__ = 10000 # this allows __rmatmul__ to work

class AlignmentBase(object):
    """Base alignment functions most alignment methods require, such as
    filtering images (such as a high pass filter) to improve the extraction of
    features needed to estimate how features have shifted from the reference
    image to the prealign image. `guess_translation` and `guess_rotation`
    allow you to start with a guess or apply a known solution. Rotation
    is *ALWAYS* applied first, then translation. This both matches MUSE's
    implementation and also how rotations and translations are recovered
    in image registration methods.
    """

    def __init__(self, prealign, reference, convolve_prealign=None,
                 convolve_reference=None, guess_translation=None,
                 guess_rotation=None, verbose=True, header=None,
                 transform_method=None, transform_method_kwargs=None,
                 filter_params=None):
        """
        Runs the base alignment prep comment to all child alignment classes.
        It will perform the initial filters needed to help with the alignment
        algorithms and will also apply and user given offsets.
        To keep track of any user given offsets, the applied offsets and
        the order they were applied are saved in `self.manually_applied_offsets`.
        When multiple image translations and rotations are stacked, the
        shifts and rotations do not necessary add linearly. To ensure the absolute
        correct total alignment parameters are correctly stack for
        the user to align the image with a rotation, then a translation,
        use the homography matrix with attribute name `self.matrix_transform`


        Parameters
        ----------
        prealign : 2d array
            Image to be aligned.
        reference : 2d array
            Reference image with the correct alignment.
        convolve_prealign : int or None, optional
            If a number, it will convolve the prealign image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        convolve_reference : int or None, optional
            If a number, it will convolve the reference image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        guess_translation : 2-array, optional
            A starting translation you want to apply before running
            the alignment. The default is None (interpreted as [0,0]).
            Positive values translate the image in the opposite
            direction. A value of [-3,-2] will translate
            the image upwards by three pixels and to the right by 2 pixels.
        guess_rotation : float, optional
            A starting rotation you want to apply before running. Units are in
            degrees, and a positive value rotates counter-clockwise.
            the alignment. The default is None (interpreted as 0).
        verbose : bool, optional
            Tells the class whether to print out information for the user.
            The default is True.
        header : header: astropy.io.Header.header
            Dictionary-like object containing the world coordinate reference.
            The default is None.
        transform_method: function
            A method for warping and image to a new grid. Default is None.
        filter_params : dict, optional
            A dictionary containing user defined parameters for the image filtering
            If the default is {}, uses the filter parameters stored in `pcc_params`

        Returns
        -------
        None.

        """

        self.verbose = verbose
        self.print = VerbosePrints(self.verbose)
        self.header = header

        self.prealign = prealign
        self.reference = reference

        self.convolve_prealign = convolve_prealign
        self.convolve_reference = convolve_reference

        if filter_params is None:
            self.filter_params = pcc_params
            self.print.default_filter_params()
        else:
            self.filter_params = filter_params

        self.prealign_filter = filter_image_for_analysis(image=self.prealign,
                                                         convolve=self.convolve_prealign,
                                                         **self.filter_params)

        self.reference_filter = filter_image_for_analysis(image=self.reference,
                                                          convolve=self.convolve_reference,
                                                          **self.filter_params)

        # This is the euclid homography matrix. It is another way
        # of storing the translation and rotation. Useful since if
        # you apply a rotation and then a translation, it will accurately track
        # the total rotation, then translation that has been applied
        #
        self.matrix_transform = HomoMatrix()
        self._update_transformation_matrix(guess_rotation, guess_translation)

        # Applying a guess/known translation and or rotation
        if transform_method is not None:
            self.transform_method = transform_method
            if transform_method_kwargs is None:
                self.transform_method_kwargs = {}
            else:
                self.transform_method_kwargs = transform_method_kwargs
        else:
            if self.header is not None:
                print("INFO = Using *reproject* to transform images")
                self.transform_method = self.apply_transform_using_reproject
                self.transform_method_kwargs = {}
            else:
                print("INFO = Using *ndimage* to transform images")
                self.transform_method = self.apply_transform
                self.transform_method_kwargs = {}

    @property
    def translation(self):
        return self.matrix_transform.translation

    @property
    def rotation_deg(self):
        return self.matrix_transform.rotation_deg

    @property
    def rotation_rad(self):
        return self.matrix_transform.rotation_rad

    @classmethod
    def from_fits(cls, filename_prealign, filename_reference, hdu_index_prealign=0,
                  hdu_index_reference=0, convolve_prealign=None, convolve_reference=None,
                  guess_translation=None, guess_rotation=None, verbose=True,
                  transform_method=None, transform_method_kwargs=None, filter_params=None):
        """
        Initialises the class straight from the filepaths

        Parameters
        ----------
        cls : object
            object equivalent to self.
        filename_prealign : str
            Filepath to the prealign fits file image.
        filename_reference : str
            Filepath gh repo clone emsellem/spacepylotto the reference fits file image.
        hdu_index_prealign : int or str, optional
            Index or dict name for prealign image if the hdu object has
            multiple objects. The default is 0.
        hdu_index_reference : int or str, optional
            Index or dict name for reference image if the hdu object has
            multiple objects. The default is 0.
        convolve_prealign : int or None, optional
            If a number, it will convolve the prealign image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        convolve_reference : int or None, optional
            If a number, it will convolve the reference image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            The default is None.
        guess_translation : 2-array, optional
            A starting translation you want to apply before running
            the alignment. Positive values translate the image in the opposite
            direction. A value of [-3,-2] will translate the image upwards by
            the image upwards by three pixels and to the right by 2 pixels.
        guess_rotation : float, optional
            A starting rotation you want to apply before running. Units are in
            degrees, and a positive value rotates counter-clockwise.
            The default is None (interpreted as 0).
        verbose : bool, optional
            Tells the class whether to print out information for the user.
            The default is True.
        transform_method : function
            A method for warping and image to a new grid. Default is None.
        transform_method_kwargs : dict, optional
        filter_params : dict, optional
            A dictionary containing user defined parameters for the image filtering
            If the default is {}, uses the filter parameters stored in `pcc_params`

        Returns
        -------
        object
            Initialises the class.

        """

        data_prealign, header = au.open_fits(filename_prealign, hdu_index_prealign)
        data_reference = au.open_fits(filename_reference, hdu_index_reference)[0]

        return cls(data_prealign, data_reference, convolve_prealign,
                   convolve_reference, guess_translation,
                   guess_rotation, verbose, header, transform_method,
                   transform_method_kwargs, filter_params)

    def _update_transformation_matrix(self, new_rotation=None, new_translation=None,
                                      new_homography_matrix=None):
        """
        Updates the transformation matrix to the new solution found

        Parameters
        ----------
        new_rotation : int, optional
            The new rotation. Units are in degrees, and a positive value rotates
            counter-clockwise. The default is 0.
        new_translation : 2-array, optional
            The new translation.  Positive values translate the image in the
            opposite direction. A value of [-3,-2] will translate the image
            upwards by three pixels and to the right by 2 pixels.
            The default is [0,0].

        Returns
        -------
        None

        """
        # If no homography matrix is provided we create it using the
        # new rotation and new translation parameters.
        # We have to change the sign of the translation, otherwise
        # it will be wrong
        if new_homography_matrix is None:
            if new_translation is None:
                new_translation = [0., 0.]
            if new_rotation is None:
                new_rotation = 0.
            new_homography_matrix = au.create_euclid_homography_matrix(
                new_rotation, np.array(new_translation), rotation_first=True)

        self.matrix_transform =  new_homography_matrix @ self.matrix_transform#self.matrix_transform.adot(new_homography_matrix)

    def apply_transform_using_reproject(self, image, rotation=0, yx_offset=None):
        """Transforms an image using the given rotation and offsets using reproject

        Parameters
        ----------
        image : 2d array
            image to be transformed
        rotation : int, optional
            The new rotation. Units are in degrees, and a positive value rotates
            counter-clockwise. The default is 0.
        yx_offset : 2-array
            The yx coordinate to translate the image. The default is [0,0]

        Returns
        -------
        image_transformed : 2d array
            image transformed using given rotation and translation
        """

        if yx_offset is None:
            yx_offset = np.zeros(2)
        transformed_image = au.transform_image_wcs(image, self.header,
                                                   rotation, yx_offset)
        self.print.applied_rotation(rotation)
        self.print.applied_translation(yx_offset)

        return transformed_image

    def apply_transform(self, image, rotation=0, yx_offset=None):
        """Transforms an image using the given rotation and offsets using
        skimage and scipy functions

        Parameters
        ----------
        image : 2d array
            image to be transformed
        rotation : int, optional
            The new rotation. Units are in degrees, and a positive value rotates
            counter-clockwise. The default is 0.
        yx_offset : 2-array
            The yx coordinate to translate the image. The default is [0,0]

        Returns
        -------
        image_transformed : 2d array
            image transformed using given rotation and translation
        """

        image_transformed = cp.copy(image)
        if rotation != 0:
            image_transformed = au.rotate_image(image_transformed, rotation)
            self.print.applied_rotation(rotation)

        # Looking at the sum of offset. Taking the square to avoid negatives.
        if yx_offset is None:
            yx_offset = np.zeros(2)
        if sum(yx_offset**2) != 0:
            image_transformed = au.translate_image(image_transformed, yx_offset)
            self.print.applied_translation(yx_offset)

        return image_transformed

    def update_prealign(self, method=None, **kwargs):
        """
        Warps the given image using the current alignment
        solution. `Method` determines what method is used to warp the image

        Parameters
        ----------
        method : func, optional
            method used for image warping. The default is None.
        **kwargs : `method` properties, optional
            Allows additional parameters to be given to the image warp method

        Returns
        -------
        transformed_image: 2d array
            Warped image (i.e., rotated and translated image)
        """

        transformed_image = cp.copy(self.prealign)
        if method is None:
            method = self.transform_method
            kwargs = self.transform_method_kwargs

        yx_offset = self.translation
        rotation = self.rotation_deg

        if rotation != 0 or sum(yx_offset**2) != 0:

            try:
                transformed_image = method(transformed_image, rotation,
                                           yx_offset, **kwargs)

            except ValueError:
                transformed_image = transformed_image.byteswap().newbyteorder()
                transformed_image = method(transformed_image, rotation,
                                           yx_offset, **kwargs)

        transformed_image_filtered = filter_image_for_analysis(image=transformed_image,
                                                               convolve=self.convolve_prealign,
                                                               **self.filter_params)
        return transformed_image_filtered


class AlignmentCrossCorrelate(AlignmentBase):
    """
    Finds the translational offset between the reference and prealign images
    using phase cross correlation. Phase cross correlation works by converting
    the images into fourier space. Here, translational offsets are represented
    as a phase difference, where there is no signal when they are out of phase,
    and a sharp peak when the image is in phase. This sharp peak is used
    to work out the offset between two images.
    """

    def fft_phase_correlation(self, prealign, reference, resolution=1000,
                              **kwargs):
        """
        Method for performing the cross correlation between two images.

        Parameters
        ----------
        prealign : 2d array
            Image to be aligned.
        reference : 2d array
            Reference image with the correct alignment.
        resolution : int, optional
            Determines how precise the algorthm will run. For example, 10 will
            find a solution closest to the first decimal place, 100 the second
            etc. The default is 1000.
        **kwargs : skimage.registration.phase_cross_correlation properties, optional
            kwargs are optional parameters for
            skimage.registration.phase_cross_correlation

        Returns
        -------
        shifts : 2-array
            The offset needed to align the prealign image with the
            reference image.  Positive values shift the prealign in the negative
            direction i.e A coordinate of 3,3, with an offset of 1,1
            will be shifted to 2,2. If the image have been prepared properly
            the returned shifts encode rotational offsets and scale changes
            between the two images.

        """

        upsample_factor = kwargs.pop('upsample_factor', resolution)

        if reference.shape != prealign.shape:
            print("ERROR: input image shapes are not the same"
                  f" reference has {reference.shape} "
                  f" prealign has {prealign.shape}")
        shifts, error, phasediff = phase_cross_correlation(
            reference_image=reference,
            moving_image=prealign,
            upsample_factor=upsample_factor, **kwargs
            )
        return shifts


class AlignTranslationPCC(AlignmentCrossCorrelate):
    """
    Class used to find only the translational offset between a pair of
    images using phase cross correlation.
    """

    def get_translation(self, split_image=1, over_shifted=20, resolution=1000,
                        **kwargs):
        """
        Method to perform phase cross correlation to find translational offset
        between two images of equal size. Artifacts can affect the performance
        of this method, therefore to make this method more robust,
        the images can be split into quarters (2) 9ths (3) etc, and
        the method is run independently in each section. Any offset found
        that has a truly bad solution (as defined by the `over_shifted`
        parameter (units of pixels), are ignored. Final solution returns
        the median solution of all valid sections.

        Parameters
        ----------
        split_image : int, optional
            The number of subsections used to find the offset solution.
            1 uses the original image, 2 splits the image 2x2=4 times,
            3  is 3x3=9 times etc. The default is 1.
        over_shifted : float or int, optional
            Defines the limit on what is considered a plausible offset
            to find in either the x or y directions (in number of pixels).
            Any offset in x or y that is larger than this value are ignored.
            The default is 20.
        resolution : int, optional
            Determines how precise the algorithm will run. For example, 10 will
            find a solution closest to the first decimal place, 100 the second
            etc. The default is 1000.
        **kwargs : skimage.registration.phase_cross_correlation properties, optional
            kwargs are optional parameters for
            skimage.registration.phase_cross_correlation

        Returns
        -------
        shifts : 2-array
            The recovered shifts, in pixels.

        """

        if self.verbose:
            print(f'\nSplitting up image into {split_image}x{split_image} '
                  f'parts and returning median offset found for all panels.')
            print(f'A pixel shift > {over_shifted:.2f} pixels in either the x '
                  f'or the y direction will be ignored from the final offset calculation.')

        all_shifts = []
        prealign_filter_transformed = self.update_prealign()
        for i in range(split_image):
            for j in range(split_image):
                split_prealign, split_reference = self._get_split_images(prealign_filter_transformed, 
                                                                         split_image, (i, j))
                shifts = self.fft_phase_correlation(split_prealign, split_reference, resolution, 
                                                    **kwargs)

                # Ignores pixels translations that are too large
                if any(np.abs(shifts) > over_shifted):
                    continue
                else:
                    all_shifts.append(shifts)

        # Taking the median - first rejecting outliers
        if len(all_shifts) < 2:
            shifts = np.array([0., 0.])
        else:
            shifts = np.nanmedian(all_shifts, axis=0)

        # Printing out the result
        self.print.get_translation(shifts, self.translation)

        # Update the matrix
        self._update_transformation_matrix(new_rotation=0.,
                                           new_translation=shifts)

        # Now getting the full shifts
        shifts = self.translation

        return shifts

    def _get_split_images(self, prealign_filter, split_image, nums):
        """
        Returns subsection of a given image split into smaller sections

        Parameters
        ----------
        split_image : int
            The number of subsections used to find the offset solution.
            1 uses the original image, 2 splits the image 2x2=4 times,
            3  is 3x3=9 times etc. The default is 1.
        nums : 2-array of ints
            the ith and jth subsection location.

        Returns
        -------
        split_pre : 2darray
            the ith, jth subsection of the prealigned image
        split_ref : 2darray
            the corresponding ith, jth subsection of the reference image.

        """

        split_size = np.array(np.shape(self.reference_filter)) / split_image

        [lower_y, upper_y], [lower_x, upper_x] = [_split(split_size[n], nums[n])
                                                  for n in [0, 1]]

        split_pre = prealign_filter[lower_y:upper_y, lower_x:upper_x]
        split_ref = self.reference_filter[lower_y:upper_y, lower_x:upper_x]

        return split_pre, split_ref


class AlignHomography(object):
    """Finds the offset between two images by estimating the matrix transform needed
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

    """

    def __init__(self, homography_matrix=None):
        """Initialisation of the Homography class
        """
        self.homography_matrix = homography_matrix

    def homography_on_grid_points(self, original_xy, transformed_xy,
                                  method=transform.EuclideanTransform,
                                  reverse_order=None,
                                  min_samples=3, residual_threshold=0.5,
                                  max_trials=1000, **kwargs):
        """Performs a fit to estimate the homography matrix that represents the grid
        transformation. Uses ransac to robustly eliminate vector outliers that
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
        min_samples : int
        residual_threshold : float
        max_trials : int
        reverse_order : bool, optional
            Homography matrix outputs the solution representing a rotation followed
            by a translation. If you want the solution to translate, then rotate
            set this to True. The default is None (and will thus use the default).

        Returns
        -------
        self.homography_matrix : 3x3 matrix
            The Matrix that has been "performed" that resulted in the offset
            between the prealign and the reference. The inverse therefore
            describes the parameters needed to convert the prealign image
            to the reference grid.

        """
        self.homography_matrix = HomoMatrix(reverse_order=reverse_order)

        # Call to Ransac
        self.model_robust, inliers = ransac((original_xy, transformed_xy), method,
                                            min_samples=min_samples,
                                            residual_threshold=residual_threshold,
                                            max_trials=max_trials, **kwargs)

        homographic_solution = np.linalg.inv(self.model_robust.params)
        homographic_solution[1, 0] = -homographic_solution[1, 0]
        homographic_solution[0, 1] = -homographic_solution[0, 1]

        # self.homography_matrix.homo_matrix = self.model_robust.params # BUG
        self.homography_matrix.homo_matrix = homographic_solution

class AlignOpticalFlow(AlignmentBase, AlignHomography):
    """Optical flow is a gradient based method that find small changes between
    two images as vectors for each pixel. Recommended for translations and
    rotations that are less than 5 pixels. If larger than this, use
    the iterative method.

    To do optical flow, the intensities between the two image much be as equal
    as possible. Changes in intensity are kinda interpreted as a scale change.
    To match the intensities, histogram equalisation, or some intensity
    scaling is needed.
    """

#     def __init__(self, prealign, reference, convolve_prealign=None,
#                  convolve_reference=None, guess_translation=None,
#                  guess_rotation=None, verbose=True, header=None,
#                  transform_method=None, transform_method_kwargs=None,
#                  filter_params=None):
#         super().__init__(prealign, reference, convolve_prealign, 
#                          convolve_reference, guess_translation, 
#                          guess_rotation, verbose, header, 
#                          transform_method, transform_method_kwargs, filter_params)

    def optical_flow(self, oflow_test=False, **kwargs):
        """
        Performs the optical flow.

        Parameters
        ----------
        oflow_test: bool
            Boolean to use the test values for the optical_flow. If False,
            it will revert to code defined defaults values set up in the
            default_kw dictionary. Default to True.

        Returns
        -------
        v : 2d array
            y component of the vectors describing the offset between the prealign
            and the reference image.
        u : 2d array
            x component of the vectors describing the offset between the prealign
            and the reference image.

        """
        if oflow_test:
            kwargs_of = {}
            print("WARNING: Optical Flow will run with test parameters values")
        else:
            kwargs_of = default_kw_opticalflow
            # Overwriting the keywords in case those are provided
            for key in kwargs:
                kwargs_of[key] = kwargs.get(key)

        prealign_filter_transformed = self.update_prealign()
        v, u = optical_flow_tvl1(self.reference_filter, prealign_filter_transformed,
                                 **kwargs_of)

        return v, u

    def _prep_oflow_for_homography(self, num_per_dimension, **kwargs_of):
        """Gets a sample of coordinates from the optical flow to perform
        homography on. A sample is needed because the full image is expensive
        to run.

        Parameters
        ----------
         num_per_dimension : int
             For each axis, this sets the number of grid points to be calculated.

        Returns
        -------
        xy_stacked : num_per_dimension x2 array
            The paired xy coordinates of the vector origin.
        xy_stacked_pre : num_per_dimension x2 array
            The paired xy coordinates of the vector end.

        """
        # Vector field. V are the y component of the vectors, u are the x components
        self.v, self.u = self.optical_flow(**kwargs_of)

        y_sparse, x_sparse, v_sparse, u_sparse = \
            au.get_sparse_vector_grid(self.v, self.u, num_per_dimension)

        y_sparse_pre = y_sparse + v_sparse
        x_sparse_pre = x_sparse + u_sparse

        xy_stacked = au.column_stack_2d(x_sparse, y_sparse)
        xy_stacked_pre = au.column_stack_2d(x_sparse_pre, y_sparse_pre)

        return xy_stacked, xy_stacked_pre

    def get_translation_rotation(self, num_per_dimension=50,
                                 homography_method=transform.EuclideanTransform,
                                 reverse_order=False, 
                                 oflow_test=False, **kwargs):
        """Works out the translation and rotation using homography once

        Parameters
        ----------
        num_per_dimension : int, optional
             For each axis, this sets the number of grid points to be calculated.
             The default is 50.
        homography_method : function, optional
            The (skimage) transformation method. Different transforms can find
            additional types of matrix transformations that might be present.
            For alignment, we only want to find rotation and transformation. For
            only these transformations, use the Euclidean transform. The default
            is transform.EuclideanTransform.
        reverse_order : bool, optional
            If the order that a user will use to  correct the alignment is NOT
            rotation, then translation, set this to True. The default is False.
        oflow_test: bool
            When True (default), using test values to speed things up.
            If False, it will use more optimal values but that will slow things down
            significantly.
        **kwargs: additional arguments
            Those arguments will be passed on to ransac in the homography
            calculation.

        Attributes
        ----------
        self.homography_matrix : 3x3 matrix
            The Matrix that has been "performed" that resulted in the offset
            between the prealign and the reference. The inverse therefore
            describes the parameters needed to convert the prealign image
            to the reference grid.

        """

        # Get keywords from kwargs to pass on to the optical flow
        kwargs_of = {'oflow_test': oflow_test}

        # Overwriting the keywords in case those are provided
        # This uses the default_kw_of list of keys
        for key in default_kw_opticalflow:
            if key in kwargs:
                kwargs_of[key] = kwargs.pop(key)

        reference_grid_xy, prealign_grid_xy = self._prep_oflow_for_homography(num_per_dimension, **kwargs_of)

        # Using homography
        # Note: method must be a method that outputs the 3x3 homography matrix
        self.homography_on_grid_points(original_xy=reference_grid_xy,
                                       transformed_xy=prealign_grid_xy,
                                       method=homography_method,
                                       reverse_order=reverse_order,
                                       **kwargs)

        # Printing and updating
        self.print.get_rotation(self.homography_matrix.rotation_deg, 
                                self.matrix_transform.rotation_deg)
        self.print.get_translation(self.homography_matrix.translation, 
                                   self.matrix_transform.translation)

        self._update_transformation_matrix(new_homography_matrix=self.homography_matrix)

    def get_iterate_translation_rotation(self, nruns_opticalflow=1, num_per_dimension=50,
                                         homography_method=transform.EuclideanTransform,
                                         reverse_order=False,
                                         oflow_test=True, **kwargs):
        """If the solution is offset by more than ~5 pixels, optical flow struggles
        to match up the pixels. This method gets around this by applying optical
        flow multiple times to that the shifts and rotations converge to a solution.
        TODO = Make this stop after a change between solutions is smaller that
        a user defined value (i.e >1% for example)

        Parameters
        ----------
        nruns_opticalflow : int, optional
            The number of times to perform optical flow to find the shifts and
            rotation. The default is 1, as normally we should use iterations
            within optical flow itself, and not from this stage.
        num_per_dimension : int, optional
             For each axis, this sets the number of grid points to be calculated.
             The default is 50.
        homography_method : function, optional
            The (skimage) transformation method. Different transforms can find
            additional types of matrix transformations that might be present.
            For alignment, we only want to find rotation and transformation. For
            only these transformations, use the Euclidean transform. The default
            is transform.EuclideanTransform.
        reverse_order : bool, optional
            If the order that a user will use to  correct the alignment is NOT
            rotation, then translation, set this to True. The default is False.
        oflow_test: bool
            If set to True (default), will use set of parameters for optical flow
            which makes it fast (but not necessarily accurate).
        **kwargs: additional arguments
            Those arguments will be passed on to ransac in the homography
            calculation.

        Attributes
        -----------
        self.matrix_transform
        self.homography_matrix
        self.translation : 2-array
            The recovered shifts, in pixels.
        self.rotation_deg : float
            The recovered rotation, in degrees.

        """
        print("Starting optical flow iterations ...")
        for i in range(nruns_opticalflow):
            print(f"Iteration #{nruns_opticalflow:02d}", end='\r')
            self.get_translation_rotation(num_per_dimension, homography_method,
                                          reverse_order,
                                          oflow_test=oflow_test, **kwargs)
        print("\n Done")
