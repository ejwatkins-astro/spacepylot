# -*- coding: utf-8 -*-

from . import alignment as align
from . import plotting as pl
from . import alignment_utilities as au
import numpy as np
import matplotlib.pyplot as plt
import copy

from astropy import wcs
from astropy.table import Table


def apply_homography_to_vector_field(homography_matrix, vector_array_2d_mapped):
    """Given a vector field, where each pixel has a vector direction,
    this transforms the vector field by the given homography matrix.
    NB: This will be incorrect if rotation is present. To apply a rotation
    to a vector field you have to apply the rotation on both the position
    vectors and the vector field values in some way.

    Parameters
    ----------
    homography_matrix : 3x3 numpy array
        The transformation matrix to apply to the vector field.
    vector_array_2d_mapped : n x m x 3 numpy array
        A 2d array where each yx position is a 3-vector of x, y, 1.

    Returns
    -------
    transformed_vector_array_2d_mapped : n x m x 3 numpy array
        A 2d array where each yx position is a 3-vector of x, y, 1 that has
        had the transformation matrix applied to it.

    """
    # vectorised matrix multiplication
    transformed_vector_array_2d_mapped = np.einsum('ij,klj->kli', homography_matrix, vector_array_2d_mapped)

    transformed_vector_array_2d_mapped = transformed_vector_array_2d_mapped[:, :, :2]

    return transformed_vector_array_2d_mapped


def get_vector_magnitude(vector_field):
    """Gets the magnitude of the vectors in a vector field.
    TODO: `pl.get_vector_norm` does the same thing but is the
    non-vectorised version. Might want to make this a wrapper
    of `pl.get_vector_norm` by getting or the v and u vectors
    separately and then call `pl.get_vector_norm`, or make
    `pl.get_vector_norm` a wrapper of this function.


    Parameters
    ----------
    vector_field : n x m x 3 numpy array
        A 2d array where each y x position is a 3-vector of x, y, 1 containing
        the residual vector remaining ager a homographic solution has been applied

    Returns
    -------
    vector_magnitude_offset : nxm numpy array
        Magnitudes of the vector field.

    """

    vector_magnitude_map = np.linalg.norm(vector_field, axis=2)
    return vector_magnitude_map


def get_residuals_from_homographic_solution(homography_matrix, vector_array_2d_mapped):
    """Wrapper function to get magnitude of the residuals of a vector
    field after a homographic solution has been applied


    Parameters
    ----------
    homography_matrix : 3x3 numpy array
        The homographic solution as a matrix that will be applied to
        the vector field.
    vector_array_2d_mapped : n x m x 3 numpy array
        A 2d array where each yx position is a 3-vector of x, y, 1.

    Returns
    -------
    residual_magnitude : nxm numpy array
        Magnitudes of the vector field.

    """

    residual_vectors = apply_homography_to_vector_field(homography_matrix, vector_array_2d_mapped)
    residual_magnitude = get_vector_magnitude(residual_vectors)

    return residual_magnitude


def get_manual_offsets(offset_table_filepath, header, index):
    """Gets the manual/given offsets (rotation, translation) needed to correct
    an alignment pointing that are in an offset table.


    Parameters
    ----------
    offset_table_filepath : str
        Path and filename to offset table.
    header : header
    index : int
        The position of the pointing in the table

    Returns
    -------
    offset_values : list of int and 2-list
        The rotational offset (degrees) and paired dec and ra offset (pixels).

    """

    imawcs = wcs.WCS(header)
    imacrval = imawcs.pixel_to_world(imawcs.wcs.crval[0], imawcs.wcs.crval[1])
    scalepix = wcs.utils.proj_plane_pixel_scales(imawcs)

    offset_table = Table.read(offset_table_filepath, format='fits')
    offsets = offset_table[index]

    ra = offsets['RA_OFFSET']
    dec = offsets['DEC_OFFSET']
    rot = offsets['ROTANGLE']

    # minus because scalepix[0] should be negative but isn't
    rapix = -ra * np.cos(np.deg2rad(imacrval.dec.value)) / scalepix[0]
    decpix = dec / scalepix[1]

    # Formatting offset values as in alignment
    offset_values = [-rot, [-decpix, -rapix]]

    return offset_values


def plot_hist_of_magnitudes(data_2d, title, label, plot_statistics=True,
                            num=-1, **kwargs):
    """Plot histogram of residual vector magnitude of a homographic solution

    Parameters
    ----------
    data_2d : nxm array
        Data to make a histogram out of
    title : str
        Name of the figure.
    label : str
        Label of the histogram.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to plot upon. The default is None.
    plot_statistics : Bool, optional
        Bool for over-plotting the distribution statistics. The default is True.
    num : int, optional
        Numbering for the automatic colouring. -1 uses the colour from the most
        recent patch object added. The default is -1.
    """
    plt.figure(title, figsize=(8, 5.5))
    plt.hist(data_2d.flatten(), label=label, **kwargs)

    plt.xlabel('Magnitude vector offset')
    plt.ylabel('Amount')
    plt.title(title)

    if plot_statistics:
        mean_data, median_data, std_data, std_perc, iqr = (get_stat_ranges(data_2d))
        plot_stats(mean_data, median_data, std_perc, label_suffix=label, num=num)

    plt.legend(loc='best')
    plt.show()


def plot_stats(mean_data, median_data, std_perc, ax=None, label_suffix=None, num=-1):
    """Plot statistics from the offsets

    Parameters
    ----------
    mean_data : float
        Mean of the data.
    median_data : float
        Median of the data.
    std_perc : 2-list of floats or float
        Standard deviation range of the data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to plot upon. The default is None.
    label_suffix : str, optional
        Additional label to add to the statistics. The default is None.
    num: int
        Numbering for the automatic colouring. -1 uses the colour from the most
        recent patch object added. The default is -1.
    """
    if ax is None:
        ax = plt.gca()
    if label_suffix is None:
        label_suffix = ''
    else:
        label_suffix = ' ' + label_suffix

    if len(std_perc) == 1:
        std_perc = [mean_data-std_perc/2, mean_data+std_perc/2]

    color = np.array(ax.patches[num].get_facecolor())
    ylims = ax.get_ylim()
    color[-1] = 1
    plt.vlines(mean_data, *ylims, colors=color, lw=2, label='Mean%s: %.2f' %(label_suffix, mean_data))
    dark_color = color/2
    dark_color[-1] = 1
    plt.vlines(median_data, *ylims, colors=dark_color, linestyles='--', lw=2, label='Median%s: %.2f' %(label_suffix, median_data))
    plt.fill_between(std_perc, *ylims, color='dimgrey', alpha=0.4, ec='k', label='Spread%s: %.2f–%.2f' %(label_suffix, *std_perc))
    ax.set_ylim(*ylims)


def get_stat_ranges(data2d):
    """Gets the statistics from the given data

    Parameters
    ----------
    data2d : nxm ndarray
        The data to derive statistics from.

    Returns
    -------
    mean_data : float
        Mean of the data.
    median_data : float
        Median of the data.
    std_data : float
        Standard deviation of the data.
    std_perc : 2-list of floats
        Percentiles at the 16th and 84th positions.
    iqr : 2-list of floats
        Interquartile range.

    """
    data = data2d.flatten()
    stats = np.nanpercentile(data, [16, 84, 25, 75, 50])
    mean_data = np.nanmean(data)

    std_data = np.nanstd(data)

    std_perc = stats[:2]
    iqr = stats[2:4]
    median_data = stats[4]

    return mean_data, median_data, std_data, std_perc, iqr


def map_vector_field(u, v):
    """Takes the u and v vectors and combines them into 1 array

    Parameters
    ----------
    u : nxm 2darray
        Vector component in the x direction.
    v : nxm 2darray
        Vector component in the y direction.

    Returns
    -------
    vector_array_2d_mapped : n x m x 3 3darray
        2d map where each pixel contains is a vector containing
        the u and v components

    """
    vector_array_2d_mapped = np.ones([*u.shape, 3])
    vector_array_2d_mapped[:, :, 0] = u[:, :]
    vector_array_2d_mapped[:, :, 1] = v[:, :]

    return vector_array_2d_mapped


def get_homography_format(info, remove_rotation=False):
    """A wrapper function so that both spacepylot conventions and np.arrays
    work.
    Takes the given alignment information and outputs a homography matrix
    of it as a numpy array


    Parameters
    ----------
    info : List containing float and 2 list, spacepylot.alignment.HomoMatrix or
    3x3 ndarray
        Contains the offset information. If a list, is should be the rotation
        in degrees, and then a list of the yx offset i.e., `[0.2 [-3, 2]]`.
    remove_rotation : bool, optional
        Removes the rotation solution. The default is False.

    Returns
    -------
    homography : 3x3 ndarray
        3x3 array containing the offset information in a matrix.

    """
    try:
        if isinstance(info, align.HomoMatrix):
           homography = copy.copy(info.homo_matrix)
#           homography[:2, -1] *= -1
#           homography[0, 1] *= -1
#           homography[1, 0] *= -1
        elif len(info[1]) == 2:
            homography = au.create_euclid_homography_matrix(*info)
        else:
            homography = copy.copy(info)
    except AttributeError:
        if len(info[1]) == 2:
            homography = au.create_euclid_homography_matrix(*info)
        else:
            homography = copy.copy(info)

    if remove_rotation:
        homography[:2, :2] = np.identity(2)

    return homography


def get_vectors_after_rotation(op_obj, compare_param):
    """Applying the rotation solution to result in u v vectors that just describe
    the remaining translation offset.

    Parameters
    ----------
    op_obj : alignment.AlignOpticalFlow object
        object containing alignment
    compare_param : List containing float and 2 list,
        spacepylot.alignment.HomoMatrix or 3x3 ndarray
        Contains the offset information. If a list, is should be the rotation
        in degrees, and then a list of the yx offset i.e., `[0.2 [-3, 2]]`.

    Returns
    -------
    u : n x m 2darray
        Vector component in the x direction.
    v : n x m 2darray
        Vector component in the y direction.

    """
    op_obj_cp = copy.deepcopy(op_obj)
    op_obj_cp.verbose=False
    homography_matrix = get_homography_format(copy.deepcopy(compare_param))
    homography_matrix[:2, -1] = 0

    try: #
        op_obj_cp.matrix_transform.homo_matrix = homography_matrix
    except AttributeError:
        op_obj_cp.matrix_transform = homography_matrix

    v, u = op_obj_cp.optical_flow()

    return u, v


class MagnitudeOffset(object):
    """If we assume optical flow finds the true offset of every pixel, we can
    check how good an alignment solution is by applying th alignment to the
    optical flow vector field. The remaining vectors and their magnitude
    are like residuals. We can get the distribution and statistics of
    these residual magnitudes and use them to quantify the alignment solution
    This object takes in two alignment solutions and compares them.

    NB: This method should run optical flow once, then apply the alignment
    solution. However, to do this, when the vectors can have rotation, it requires
    differnt vector maths (need to rotate bothe the position
    vectors and the vector field in some way.) Instead, we subtract the rotation
    from the solution by applying it to the prealign image, allowing simple
    addition of the translation solution to get the residuals. This results
    in two optical flow solutions as a result.
    TODO: Either figure out the vector maths, or simplify how translation is
    applied (Currently  the addition is apllied using vector multiplication,
    but it can be simplified massivly now the rotation is subtracted at a
    different stage.)
    The methods

    """
    def __init__(self, u1, v1, u2, v2, compare_param_1, compare_param_2, header=None):
        """Initialise using the two vector solutions with rotation
        already subtracted.


        Parameters
        ----------
        u1 : nxm ndarray
            x component of vector field for solution 1 with the rotation subtracted.
        v1 : nxm ndarray
            y component of vector field for solution 1 with the rotation subtracted.
        u2 : nxm ndarray
            x component of vector field for solution 2 with the rotation subtracted.
        v2 : nxm ndarray
            y component of vector field for solution 2 with the rotation subtracted.
        compare_param_1 :List containing float and 2 list,
            spacepylot.alignment.HomoMatrix or 3x3 ndarray
            Contains the offset information for the first solution.
            If a list, is should be the rotation in degrees, and then
            a list of the yx offset i.e., `[0.2 [-3, 2]]`. Assumes
            this is to be the manual solution for plotting. If not
            just change the labels when plotting
        compare_param_2 :List containing float and 2 list,
            spacepylot.alignment.HomoMatrix or 3x3 ndarray
            Contains the offset information for the first solution.
            If a list, is should be the rotation in degrees, and then
            a list of the yx offset i.e., `[0.2 [-3, 2]]`.
        header: astropy.io.Header.header
            Dictionary-like object containing the world coordinate reference.

        Returns
        -------
        None.

        """
        self.u1 = u1
        self.v1 = v1
        self.u2 = u2
        self.v2 = v2

        self.header = header
        self.compare_param_1 = compare_param_1
        self.compare_param_2 = compare_param_2

        self.vector_array_2d_mapped_1 = map_vector_field(self.u1, self.v1)
        self.vector_array_2d_mapped_2 = map_vector_field(self.u2, self.v2)

        self._get_homography()

        self.residual_magnitudes_1, self.residual_magnitudes_2 = self.get_residual_magnitudes()

    @classmethod
    def from_op_align_object(cls, op_align, compare_param_1, compare_param_2=None, header=None):
        """Initialise starting from an align object, which will contain the
        image maps, headers etc. Still needs to rerun the optical flow solution
        with the rotation removed from the prealign image


        Parameters
        ----------
        cls : cls object
            Python method
        op_align : alignment.AlignOpticalFlow object
            object containing alignment
        compare_param_1 :List containing float and 2 list,
            spacepylot.alignment.HomoMatrix or 3x3 ndarray
            Contains the offset information for the first solution.
            If a list, is should be the rotation in degrees, and then
            a list of the yx offset i.e., `[0.2 [-3, 2]]`. Assumes
            this is to be the manual solution for plotting. If not
            just change the labels when plotting
        compare_param_2 :List containing float and 2 list,
            spacepylot.alignment.HomoMatrix or 3x3 ndarray
            Contains the offset information for the second solution.
            If a list, is should be the rotation in degrees, and then
            a list of the yx offset i.e., `[0.2 [-3, 2]]`. If compare_param_2
            is not given, will use the alignment solution from the align object.
            The default is None.
        header: astropy.io.Header.header
            Dictionary-like object containing the world coordinate reference.
            The default is None.

        Returns
        -------
        MagnitudeOffset.from_op_align_object

        """
        if compare_param_2 is None:
            compare_param_2 = copy.copy(op_align.matrix_transform)

        if header is None:
            header = op_align.header

        compare_param_1 = copy.copy(compare_param_1)
        compare_param_2 = copy.copy(compare_param_2)

        u1, v1 = get_vectors_after_rotation(op_align, compare_param_1)
        u2, v2 = get_vectors_after_rotation(op_align, compare_param_2)

        return cls(u1, v1, u2, v2, compare_param_1, compare_param_2, header=header)

    @classmethod
    def from_fits(cls, filename_prealign, filename_reference,
                  compare_param_1, compare_param_2,
                  hdu_index_prealign=0, hdu_index_reference=0,
                  convolve_prealign=None, convolve_reference=None,
                  guess_translation=None, guess_rotation=None, verbose=True,
                  transform_method=None, transform_method_kwargs=None, filter_params=None):
        """Initialise starting from file names


        Parameters
        ----------
        cls : cls object
            Python method
        filename_prealign : str
            Filepath to the prealign fits file image.
        filename_reference : str
            Filepath to the reference fits file image.
        compare_param_1 :List containing float and 2 list,
            spacepylot.alignment.HomoMatrix or 3x3 ndarray
            Contains the offset information for the first solution.
            If a list, is should be the rotation in degrees, and then
            a list of the yx offset i.e., `[0.2 [-3, 2]]`. Assumes
            this is to be the manual solution for plotting. If not
            just change the labels when plotting
        compare_param_2 :List containing float and 2 list,
            spacepylot.alignment.HomoMatrix or 3x3 ndarray
            Contains the offset information for the first solution.
            If a list, is should be the rotation in degrees, and then
            a list of the yx offset i.e., `[0.2 [-3, 2]]`.
        hdu_index_prealign : int or str, optional
            Index or dict name for prealign image if the hdu object has
            multiple objects. The default is 0.
        hdu_index_reference : int or str, optional
            Index or dict name for reference image if the hdu object has
            multiple pbjects. The default is 0.
        convolve_prealign : int or None, optional
            If a number, it will convolve the prealign image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        convolve_reference : int or None, optional
            If a number, it will convolve the reference image. The number refers to
            the sigma of the folding Gaussian to convolve by in units of pixels
            (I think). The default is None.
        """
        compare_param_1 = copy.copy(compare_param_1)
        compare_param_2 = copy.copy(compare_param_2)
        op = align.AlignOpticalFlow.from_fits(
            filename_prealign=filename_prealign, filename_reference=filename_reference,
            convolve_prealign=convolve_prealign, convolve_reference=convolve_reference,
            hdu_index_prealign=hdu_index_prealign, hdu_index_reference=hdu_index_reference,
            guess_translation=guess_translation, guess_rotation=guess_rotation, verbose=verbose,
            transform_method=transform_method, transform_method_kwargs=transform_method_kwargs, 
            filter_params=filter_params)

        return MagnitudeOffset.from_op_align_object(op, compare_param_1,
                                                    compare_param_2, header=op.header)

    def _get_homography(self, remove_rotation=True):
        """Gets the alignment solution in a matrix format and removes the rotation
        solution from the matrix

        Parameters
        ----------
        remove_rotation : bool, optional
            Whether to remove the rotation information. The default is True.

        Returns
        -------
        None.

        """
        self.homography_1 = get_homography_format(self.compare_param_1, remove_rotation)
        self.homography_2 = get_homography_format(self.compare_param_2, remove_rotation)

    def get_residual_magnitudes(self):
        """Wrapper function to get the residual magnitudes

        Returns
        -------
        residual_magnitudes1 : nxm ndarray
            The residual magnitudes for the first alignment solution.
        residual_magnitudes2 : nxm ndarray
            The residual magnitudes for the first alignment solution.

        """
        residual_magnitudes1 = get_residuals_from_homographic_solution(self.homography_1,
                                                                       self.vector_array_2d_mapped_1)
        residual_magnitudes2 = get_residuals_from_homographic_solution(self.homography_2,
                                                                       self.vector_array_2d_mapped_2)

        return residual_magnitudes1, residual_magnitudes2

    def get_statistics(self):
        """Gets the statistics of the residual magnitudes for both solutions

        Returns
        -------
        stats_1 : dict
            Statistics of the residual magnitudes for the first solution.
        stats_2 : dict
            Statistics of the residual magnitudes for the second solution.

        """
        labels = ['mean', 'median', 'std', 'std_perc', 'iqr']

        stats_1 = get_stat_ranges(self.residual_magnitudes_1)
        stats_2 = get_stat_ranges(self.residual_magnitudes_2)

        stats_1 = dict(np.column_stack([labels, stats_1]))
        stats_2 = dict(np.column_stack([labels, stats_2]))

        return stats_1, stats_2

    def plot_hist(self, label_1='Manual', label_2='Auto', plot_statistics=True):
        """Plots the histogram of the residual magnitudes for both alignment
        solutions

        Parameters
        ----------
        label_1 : str, optional
            Label to identify the first alignment solution. The default is 'Manual'.
        label_2 : str, optional
            Label to identify the second alignment solution. The default is 'Auto'.
        plot_statistics : bool, optional
            Whether to plot the statistical measures the data. The default is True.
        """
        plot_hist_of_magnitudes(self.residual_magnitudes_2,
                                'Residual magnitude offset', label=label_2,
                                plot_statistics=plot_statistics, bins=100,
                                alpha=0.6)
        plot_hist_of_magnitudes(self.residual_magnitudes_1,
                                'Residual magnitude offset',
                                label=label_1,
                                plot_statistics=plot_statistics,
                                num=-1, bins=100, alpha=0.6)

    def plot_vector_fields(self, titles=None, vmin_perc=5, vmax_perc=95,
                           num_per_dimension=20, **kwargs):
        """Plots the residual vectors and their magnitude
        TODO: Change `pl.AlignmentPlotting.illustrate_vector_fields`
        so that the magnitude plotting is a separate function. We can then just
        call it here with a little more control.
        TODO: clean up this method a bit

        Parameters
        ----------
        titles : list of Str, optional
            List containing the titles of each alignment solution.
            The default is None.
        vmin_perc : float, optional
            Minimum percentile value of the intensity calculated on the
            prealign-reference. The default is 5.
        vmax_perc : float, optional
            Maximum percentile value of the intensity calculated on the
            prealign-reference. The default is 95.
        num_per_dimension : int, optional
            For each axis, this sets the vectors to show per column and row.
            The default is 20
        **kwargs : dict
            plt.imshow kwargs.

        Raises
        ------
        ValueError
            If only one title is provided, raise an error.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure canvas.
        self.axs: List of axis objects
            The axis objects of the subplots.
        cbar : matplotlib.colorbar.Colorbar
            colorbar object.

        """
        if titles is None:
            titles = ['Manual', 'Auto']
        else:
            if len(titles) != 2:
                raise ValueError('`titles` needs to contain two strings')

        pl_v = pl.AlignmentPlotting(self.u1, self.u1, header=self.header)

        # spelling of function changed in spacepylot to `initialise_fig_ax`
        fig, self.axs = pl.initialise_fig_ax(fig_name='residuals',
                                             fig_size=pl_v.fig_size,
                                             header=self.header, grid=[1, 2])
        all_data = np.append(self.residual_magnitudes_1, self.residual_magnitudes_2, axis=0)
        vmin, vmax = np.nanpercentile(all_data, [vmin_perc, vmax_perc])

        vmin = kwargs.pop('vmin', vmin)
        vmax = kwargs.pop('vmin', vmax)

        vmin, vmax = pl.plot_image(self.residual_magnitudes_1, self.axs[0], title=titles[0],
                                   return_colourange=True, vmin=vmin, vmax=vmax, **kwargs)[1:]
        pl.plot_image(self.residual_magnitudes_2, self.axs[1], title=titles[1],
                      return_colourange=False, vmin=vmin, vmax=vmax, **kwargs)

        ax = self.axs[1]
        cax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.02, ax.get_position().height])

        cbar = plt.colorbar(self.axs[1].images[0],cax=cax)
        cbar.set_label('Magnitude')

        residual_vectors_1 = apply_homography_to_vector_field(self.homography_1,
                                                              self.vector_array_2d_mapped_1)
        residual_vectors_2 = apply_homography_to_vector_field(self.homography_2,
                                                              self.vector_array_2d_mapped_2)

        pl_v.overplot_vectors(self.axs[0], v=residual_vectors_1[:,:, 1],
                              u=residual_vectors_1[:, :, 0],
                              num_per_dimension=num_per_dimension)
        pl_v.overplot_vectors(self.axs[1], v=residual_vectors_2[:,:, 1],
                              u=residual_vectors_2[:, :, 0],
                              num_per_dimension=num_per_dimension)

        return fig, self.axs, cbar
