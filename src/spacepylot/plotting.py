# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:46:37 2021

@author: Liz_J
"""
__author__ = "Elizabeth Watkins"
__copyright__ = "Elizabeth Watkins"
__license__ = "MIT License"
__contact__ = "<liz@email"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.stats import sigma_clip

from skimage import exposure

import colorcet as cc

from . import alignment_utilities as au
from . import fitsTools as fT

vector_plot_properties = {
    'headwidth': 4,
    'width': 0.003,
    'scale': 30,
    'alpha': 0.8,
    'minlength': 0.1
}

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.default'] = 'regular'


def fix_tight_layout(x_length, y_length, gridding, wspace=0, hspace=None,
                     fig_height=5):
    """Creates a figure canvas that is the correct aspect ratio so plots are perfectly
    flush

    Parameters
    ----------
    x_length : float
        Length of data to be displayed on the x axis.
    y_length : float
        Length of data to be displayed on the y axis.
    gridding : (int, int)
        The subplot grid as rows and columns.
    wspace : float, optional
        White space between subplots on the same row. The default is 0.
    hspace : float or None, optional
        White space between subplots on the same row. The default is None.
    fig_height : float, optional
        Height of the figure in inches. The default is 5.

    Returns
    -------
    fig_size : 2-tuple
        size of the figure which when used optimises the white space.
    subplots_adjust_params : dict
        parameters to input into `matplotlib.pyplot.subplots_adjust` to
        make the canvas flush.

    """
    aspect = x_length/y_length
    # n=number of rows, m = number of columns
    n, m = gridding

    bottom = 0.1
    left = 0.05
    top = 1.-bottom
    right = 1.-left

    fig_aspect = (1 - bottom - (1 - top)) / (1 - left - (1 - right))
    if hspace is None:
        hspace = wspace/float(aspect)

    fig_width = fig_height * fig_aspect * (m + (m - 1) * wspace) \
                / ((n + (n - 1) * hspace) * aspect)

    fig_size = (fig_width, fig_height)

    subplots_adjust_params = {
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
        'wspace': wspace,
        'hspace': hspace
    }

    return fig_size, subplots_adjust_params


def create_gsgrid(figure_name, plot_gridding, **kwargs):
    """Creates the subplot grid using gspec grid

    Parameters
    ----------
    figure_name : str or int
        Name used to initialise the figure
    plot_gridding : 2-tuple
        The subplot grid as rows and columns.
    **kwargs : dict
        kwargs for `matplotlib.pyplot.figure`. object

    Returns
    -------
    fig : `matplotlib.pyplot.figure`
        Initialised figure object.
    gs : gridspec.GridSpec
        Initialised subplot grid object.
    """
    fig = plt.figure(figure_name, **kwargs)
    # first number is the amount of rows, second is the amount of columns
    gs = gridspec.GridSpec(*plot_gridding)
    return fig, gs


def add_subplot(gs, subplot_pos, subplot_size=(1, 1), fig=None):
    if fig is None:
        fig = plt.gcf()

    ax = fig.add_subplot(gs[subplot_pos[0]:subplot_pos[0] + subplot_size[0],
                         subplot_pos[1]:subplot_pos[1] + subplot_size[1]])

    return ax


def initialise_fig_ax(fig_name=None, fig_size=None, header=None, grid=None):
    """
    Initialise the figure axis with x and y pixels labels and the gridding
    needed for plotting

    Parameters
    ----------
    fig_name: str
        Name of the figure
    fig_size: tuple
        Size of the figure
    header: header
    grid: tuple or list
       2 numbers to give the gridding set

    Returns
    -------
    fig: figures
    axs: axis
    """
    if grid is None:
        grid = [1, 1]
    fig, gs = create_gsgrid(figure_name=fig_name, plot_gridding=grid,
                            figsize=fig_size)

    if header is None:
        axs = [add_subplot(gs, (i//grid[1], i%grid[1]), fig=fig)
               for i in range(np.multiply(*grid))]
        axs[0].set_ylabel('y pixels')
        axs[0].set_xlabel('x pixels')
        axs[1].set_xlabel('x pixels')
    else:
        headers = [header] * grid[0] * grid[1]
        ras, decs, axs, fig = fT.subfigure_wsc(fig_name, headers, plot_gridding=grid)

    return fig, axs


def make_red_blue_overlay(red_image, blue_image):
    """
    Function converts intensities of two given images into a rbg array
    for plotting. Data must have identical dimensions

    Parameters
    ----------
    red_image : ndarray
        2d array containing the red channel image
    blue_image : ndarray
         2d array containing the blue/cyan channel image.

    Returns
    -------
    rgb_image : 3darray
        rgb intensities in third dimension with last two dimensions
        matching the input images

    """
    red_shape = red_image.shape
    blue_shape = blue_image.shape
    if blue_shape != red_shape:
        print("ERROR: the two images should have the same shape")
        return None

    rgb_image = np.zeros([*red_shape, 3])

    red_image = exposure.equalize_hist(np.nan_to_num(red_image))
    blue_image = exposure.equalize_hist(np.nan_to_num(blue_image))

    rgb_image[..., 0] = red_image
    rgb_image[..., 1] = blue_image
    rgb_image[..., 2] = blue_image

    return rgb_image


def crop_data(data, border=10):
    """Crop a 2D data and return it cropped after a border
    has been removed (number of pixels) from each edge
    (borderx2 pixels are removed from each dimension)
    
    Input
    -----
    data: 2d array
        Array which has the signal to be cropped
    border: int
        Number of pixels to be cropped at each edge
    
    Returns
    -------
    cdata: 2d array
        Cropped data array
    """
    if border <= 0:
        return data

    if data.ndim != 2:
        print("WARNING: Input data to crop is not 2, "
              "returning the original data")
        return data

    if (data.shape[0] > 2 * border) & (data.shape[1] > 2 * border):
        return data[border:-border, border:-border]
    else:
        print("WARNING: Data is not being cropped, as shape is {0} "
              " while border is {1}".format(data.shape, border))
        return data


def get_flux_range(data, low=2, high=98, border=25):
    """Get the range of fluxes within the array
    by looking at percentiles.
     
    Input
    -----
    data: 2d array
        Input array with signal to process
    low, high: two floats (10, 99)
        Percentiles to consider to filter
    
    Returns
    -------
    lperc, hperc: 2 floats
        Low and high percentiles
    """
    # Omit the border pixels
    data = crop_data(data, border)

    # Clean up the NaNs
    data = np.nan_to_num(data)
    if data.size > 0:
        lperc = np.percentile(data[data > 0.], low)
        hperc = np.percentile(data[data > 0.], high)
    else:
        lperc, hperc = 0., 1.

    return lperc, hperc


def make_contours(comp_image, ref_image, ax=None, title='', 
                  **kwargs):
    """Show the contours for two images

    Parameters
    ----------
    comp_image : 2darray
        2d array containing the red channel image
    ref_image : 2d array
         2d array containing the blue/cyan channel image.
    ax : matplotlib.pyplot ax, optional
        Matplotlib axis object. Default is None
    title : str, optional
        Sets title of the given image. The default is ''.

    Returns
    -------
    ax : matplotlib.pyplot ax, optional
        Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()

    # Defining the levels
    levels = kwargs.pop('levels', None)
    nlevels = int(kwargs.pop('nlevels', 10))

    if levels is None:
        # Find the high and low levels
        lowlevel, highlevel = get_flux_range(comp_image)
        levels = np.linspace(np.log10(lowlevel),
                             np.log10(highlevel), 
                             nlevels)

    # plotting the first contour - comparison image
    compset = ax.contour(np.log10(comp_image), levels, colors='k', 
                         origin='lower', linestyles='solid')
    # plotting the second contour - reference image
    refset = ax.contour(np.log10(ref_image), levels=levels,
                        colors='r', origin='lower', alpha=0.5,
                        linestyles='solid')
    ax.set_aspect('equal')
    h1, _ = compset.legend_elements()
    h2, _ = refset.legend_elements()
    ax.legend([h1[0], h2[0]], ['COMP', 'REF'])
    ax.set_title(title)

    return ax


def make_division(comp_image, ref_image, ax=None, title='',
                  percentage=5):
    """Show the division between the two images

    Parameters
    ----------
    comp_image : 2darray
        2d array containing the red channel image
    ref_image : 2d array
         2d array containing the blue/cyan channel image.
    ax : matplotlib.pyplot ax, optional
        Matplotlib axis object. Default is None
    title : str, optional
        Sets title of the given image. The default is ''.
    percentage : float, optional
        Default is 5. Will be used as cuts of the colour table

    Returns
    -------
    ax : matplotlib.pyplot ax, optional
        Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()

    # Show the percentage division
    ratio = 100. * (ref_image - comp_image) / (comp_image + 1.e-12)
    im = ax.imshow(ratio, vmin=-percentage, vmax=percentage)
    _ = plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_title(title)

    return ax


def plot_rgb(rgb_image, ax=None, title='', **kwargs):
    """Plots given rgb array

    Parameters
    ----------
    rgb_image : 3darray
        3d array containing RGB scaled values.
    ax : matplotlib.pyplot ax, optional
        Matplotlib axis object. Default is None
    title : str, optional
        Sets title of the given image. The default is ''.

    Returns
    -------
    ax : matplotlib.pyplot ax, optional
        Matplotlib axis object

    """
    if ax is None:
        ax = plt.gca()

    ax.imshow(rgb_image, origin='lower', **kwargs)
    ax.set_title(title)

    return ax


def plot_image(image, ax=None, title='', vmin_perc=5, vmax_perc=95,
               return_colourange=False, **kwargs):
    """Plots given rgb array

    Parameters
    ----------
    image : 2darray
        2d array of intensity values
    ax : matplotlib.pyplot ax, optional
        Matplotlib axis object. Default is None
    title : str, optional
        Sets title of the given image. The default is ''.
    vmin_perc: float
    vmax_perc: float
    return_colourange: bool

    Returns
    -------
    ax : matplotlib.pyplot ax, optional
        Matplotlib axis object
    """
    vmin, vmax = np.nanpercentile(image, [vmin_perc, vmax_perc])
    vmin = kwargs.pop('vmin', vmin)
    vmax = kwargs.pop('vmax', vmax)

    if ax is None:
        ax = plt.gca()

    ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax, **kwargs)
    ax.set_title(title)
    if return_colourange:
        return ax, vmin, vmax
    else:
        return ax


prealign_titles = {
    'red-blue': 'Prealign-red, reference-cyan',
    'prealign-vs-align': 'Prealign-reference',
    'vector-fields': 'Vector field'
}
align_titles = {
    'red-blue': r'Align-red, reference-cyan. Dy: %.3f, '
                r'Dx: %.3f, Dtheta: %.3f',
    'prealign-vs-align': r'Aligned-reference. Dy: %.3f, '
                         r'Dx: %.3f, Dtheta: %.3f',
    'vector-fields': 'Vector field average removed [y=%.2f,x=%.2f]',
    'contours': 'Comparison contours',
    'division': 'Comparison division'
}

titles = {
    'before': prealign_titles,
    'after': align_titles
}


class AlignmentPlotting:
    """Contains tools for plotting and inspecting the alignment
    """

    def __init__(self, prealign, reference, rotation=0, shifts=None,
                 header=None, v=None, u=None, rotation_fit=None):
        """Initialise the object to show the difference between the prealign and align
        compared to a reference image for a given rotation and translation.
        Rotation is applied first, then translation

        Parameters
        ----------
        prealign : 2d-array
            Image showing the prealigned image.
        reference : 2d-array
            Image showing the reference image.
        rotation : float, optional
            Rotation to apply to the prealign image to align it with the reference
            image. The default is 0.
        shifts :2-array float, optional
            Shifts to apply to the prealign image to align it with the reference
            image. The default is [0,0].
        header: astropy.io.Header.header
            Dictionary-like object containing the world coordinate reference.
        v : 2d-array, optional
            y component of the vectors describing the offset between the prealign
            and the reference image. The default is None.
        u : 2d-array, optional
            x component of the vectors describing the offset between the prealign
            and the reference image. The default is None.
            rotation_fit : TYPE, optional
        rotation_fit : dict, optional
            Dictionary parameters showing the rotational solution found using
            the iterative phase cross correlation method. The default is {}.

        Returns
        -------
        None.

        """
        if rotation is None and shifts is None:
            raise ValueError('`AlignmentPlotting` object has no registration '
                             'information. Please provide a shift and/or '
                             'rotation offset')
        self.prealign = prealign
        self.reference = reference
        self.rotation = rotation
        if shifts is None:
            self.shifts = [0., 0.]
        else:
            self.shifts = shifts
        self.header = header
        self.v = v
        self.u = u
        if rotation_fit is None:
            self.rotation_fit = {}
        else:
            self.rotation_fit = rotation_fit

        image_shape = prealign.shape

        if self.header is not None:
            self.aligned = au.transform_image_wcs(self.prealign, self.header,
                                                  self.rotation, self.shifts)
        else:
            try:
                rotation_corrected = au.rotate_image(self.prealign, self.rotation)
            except ValueError:
                self.prealign = self.prealign.byteswap().newbyteorder()
                rotation_corrected = au.rotate_image(self.prealign, self.rotation)

            self.aligned = au.translate_image(rotation_corrected, self.shifts)

        self.fig_size, self.fig_params = fix_tight_layout(image_shape[-1],
                                                          image_shape[-2],
                                                          (1, 2),
                                                          fig_height=7)

    @classmethod
    def from_fits(cls, filename_prealign, filename_reference, rotation,
                  shifts, v=None, u=None, hdu_index_prealign=0,
                  hdu_index_reference=0):
        """Initialises the alignment plotting from fits files and user input
        translation and rotational offsets

        Parameters
        ----------
        cls : obj
            object version of self
        filename_prealign : str
            Filepath to the prealign fits file image.
        filename_reference : str
            Filepath to the reference fits file image.
        rotation : float
            Rotation to apply to the prealign image in degrees Positive values
            rotate anti clockwise and negative rotate clockwise.
        shifts : 2-arraylike
            The y and x shift offset bewtween the prealign and reference image.
            Positive values shift the prealign in the negative direction i.e
            A coordinate of 3,3, with an offser of 1,1 will be shifted to 2,2.
        v : 2d-array, optional
            y component of the vectors describing the offset between the prealign
            and the reference image. The default is None.
        u : 2d-array, optional
            x component of the vectors describing the offset between the prealign
            and the reference image. The default is None.
        hdu_index_prealign : int or str, optional
            Index or dict name for prealign image if the hdu object has
            multiple pbjects. The default is 0.
        hdu_index_reference : int or str, optional
            Index or dict name for reference image if the hdu object has
            multiple pbjects. The default is 0.

        Returns
        -------
        class
            Initialises the class.

        """

        data_prealign, header = au.open_fits(filename_prealign, hdu_index_prealign)
        data_reference = au.open_fits(filename_reference, hdu_index_reference)[0]

        return cls(data_prealign, data_reference, rotation, shifts,
                   header=header, v=v, u=u)

    @classmethod
    def from_align_object(cls, align_object, rotation=0, shifts=None,
                          header=None, v=None, u=None,
                          rotation_fit=None):
        """Initialises the class given an alignment object that contains the same
        attributes needed to plot the alignment, with an alignment solution

        Parameters
        ----------
        cls : obj
            object version of self
        align_object : align object
            Given align object.
        rotation : float
            Rotation to apply to the prealign image in degrees Posative values
            rotate anti clockwise and negative rotate clockwise.
        shifts : 2-arraylike
            The y and x shift offset bewtween the prealign and reference image
            Posative values shift the prealign in the negative direction i.e
            A coordinate of 3,3, with an offser of 1,1 will be shifted to 2,2.
        header: header
            Fits header.
        v : 2d-array, optional
            y component of the vectors describing the offset between the prealign
            and the reference image. The default is None.
        u : 2d-array, optional
            x component of the vectors describing the offset between the prealign
            and the reference image. The default is None.
        rotation_fit: dict
           Dictionary parameters showing the rotational solution found using
           the iterative phase cross correlation method. The default is None.

        Returns
        -------
        class
            Initialises the class.

        """
        data_prealign = align_object.prealign
        data_reference = align_object.reference
        try:
            rotation = align_object.rotation_deg
        except AttributeError:
            pass
        try:
            shifts = align_object.translation
            toto = align_object.translation
        except AttributeError:
            if shifts is None:
                shifts = [0., 0.]
        try:
            v = align_object.v
        except AttributeError:
            pass
        try:
            u = align_object.u
        except AttributeError:
            pass
        try:
            header = align_object.header
        except AttributeError:
            pass
        try:
            rotation_fit = align_object.rotation_fit
        except AttributeError:
            if rotation_fit is None:
                rotation_fit = {}

        return cls(data_prealign, data_reference, rotation, shifts,
                   header, v, u, rotation_fit)

    def red_blue_before_after(self):
        """Plots the alignment before and after applying the offsets with
        red showing the prealign/align image and cyan showing the reference image
        Image is well aligned if the aligned image is grey scale with no visible
        red or blue ghosts. Each image has been histogram equalised to enhance
        contrast and so that the intensity is scaled to rgb colour ranges.

        Returns
        -------
        ax0 : matplotlib.pyplot ax
            Matplotlib axis object for the prealigned image
        ax1 : matplotlib.pyplot ax
            Matplotlib axis object for the aligned image
        """

        rgb_before = make_red_blue_overlay(self.prealign, self.reference)
        rgb_after = make_red_blue_overlay(self.aligned, self.reference)

        fig, axs = initialise_fig_ax(
            fig_name='red-blue',
            fig_size=self.fig_size,
            header=self.header,
            grid=[1, 2]
        )

        ax0 = plot_rgb(rgb_before, ax=axs[0], title=titles['before']['red-blue'])
        ax1 = plot_rgb(rgb_after,  ax=axs[1],
                       title=titles['after']['red-blue'] % (*self.shifts, self.rotation))

        fT.remove_overlapping_tickers_for_horizontal_subplots(1, *axs)
        [fT.minor_tickers(ax) for ax in axs]
        plt.subplots_adjust(**self.fig_params)

        return ax0, ax1

    def plot_divcontours(self, nlevels=10, levels=None, percentage=5.0):
        """Plots the alignment after the offsets by dividing the input reference
        image and the aligned one, and showing the contours.
        Offsets should show up as negative/positive residuals systematically on
        a given side of the resolved sources, or shifts in the contours.

        Returns
        -------
        ax0 : matplotlib.pyplot ax
            Matplotlib axis object for the division
        ax1 : matplotlib.pyplot ax
            Matplotlib axis object for the contours
        """

        fig, axs = initialise_fig_ax(fig_name='divcontours',
                                     fig_size=self.fig_size,
                                     header=self.header,
                                     grid=[1, 2])

        ax0 = make_division(self.aligned, self.reference, ax=axs[0], 
                            title=titles['after']['division'],
                            percentage=percentage)
        ax1 = make_contours(self.aligned, self.reference, ax=axs[1], 
                            title=titles['after']['contours'],
                            nlevels=nlevels, levels=levels)

        plt.subplots_adjust(**self.fig_params)

        return ax0, ax1

    def before_after(self, vmin_perc=1, vmax_perc=99, **kwargs):
        """
        Plots the alignment before and after applying the offsets by subtracting
        the reference image from the prealigned/aligned image

        Parameters
        ----------
        vmin_perc : float, optional
            Minimum percentile value of the intensity calculated on the
            prealign-reference. The default is 1.
        vmax_perc : float, optional
            Maximum percentile value of the intensity calculated on the
            prealign-reference. The default is 99.
        **kwargs : kwargs
            Kwargs for adjusting the imshow image.

        Returns
        -------
        ax0 : matplotlib.pyplot ax
            Matplotlib axis object for the prealigned image
        ax1 : matplotlib.pyplot ax
            Matplotlib axis object for the aligned image

        """

        before = self.prealign - self.reference
        after = self.aligned - self.reference

        fig, axs = initialise_fig_ax(
            fig_name='prealign-vs-align',
            fig_size=self.fig_size,
            header=self.header,
            grid=[1, 2]
        )

        ax0, vmin, vmax = plot_image(before, axs[0], title=titles['before']['prealign-vs-align'],
                                     vmin_perc=vmin_perc, vmax_perc=vmax_perc,
                                     return_colourange=True, **kwargs)
        ax1 = plot_image(after, axs[1],
                         title=titles['after']['prealign-vs-align']
                         % (*self.shifts, self.rotation),
                         vmin=vmin, vmax=vmax)

        fT.remove_overlapping_tickers_for_horizontal_subplots(1, *axs)
        [fT.minor_tickers(ax) for ax in axs]
        plt.subplots_adjust(**self.fig_params)

        return ax0, ax1

    def overplot_vectors(self, ax, v=None, u=None, num_per_dimension=20):
        """On a given axis, will plot the provided vector field

        Parameters
        ----------
        ax : matplotlib.pyplot ax
            Ax object to overplot vectors onto.
        v : 2d-array
            y component of the vectors describing the offset between the prealign
            and the reference image.
        u : 2d-array
            x component of the vectors describing the offset between the prealign
            and the reference image.
        num_per_dimension : int, optional
            For each axis, this sets the number of grid points to be calculated.
            The default is 20
        Raises
        ------
        ValueError
            Cannot plot vectors if none are supplied.

        Returns
        -------
        None.
        """

        if v is None or u is None:
            v = self.v
            u = self.u
        if v is None or u is None:
            raise ValueError('yx vector information has not been supplied')

        y_sparse, x_sparse, v_sparse, u_sparse = au.get_sparse_vector_grid(v, u,
                                                                           num_per_dimension)
        plot_vectors(v_sparse, u_sparse, y_sparse, x_sparse, ax)

    def illustrate_vector_fields(self, v=None, u=None, num_per_dimension=20,
                                 average_type=np.nanmean, avg_kwargs=None):
        """
        Plots the given vector field over the magnitude of the vector
        alongside the same vector field after subtracting the average
        vector. This helps to show if any residual translation or offset
        is present

        Parameters
        ----------
        v : 2d-array, optional
            y component of the vectors describing the offset between the prealign
            and the reference image. The default is None.
        u : 2d-array, optional
            x component of the vectors describing the offset between the prealign
            and the reference image. The default is None.
        num_per_dimension : int, optional
            For each axis, this sets the number of grid points to be calcualted.
            The default is 20
        average_type : bool or callable function
            If False, np.nanmean` is used, True, np.nanmedian is used. But
            user can also enter their own callable function to calculate the
            "average" vector". Must be able to work with nd arrays. The default
            is `np.nanmean`

        Raises
        ------
        ValueError
            Cannot plot vectors if none are supplied.

        Returns
        -------
        None.

        """

        if v is None or u is None:
            v = self.v
            u = self.u
        if v is None or u is None:
            raise ValueError('yx vector information has not been supplied')

        if avg_kwargs is None:
            avg_kwargs = {}
        v_mean, u_mean = average_flow_vector(v, u, average_type, **avg_kwargs)
        v_centred = v-v_mean
        u_centred = u-u_mean

        vector_magnitude_centred = get_vector_norm(v_centred, u_centred)
        vector_magnitude = get_vector_norm(v, u)

        fig, axs = initialise_fig_ax(
            fig_name='vector-fields',
            fig_size=self.fig_size,
            header=self.header,
            grid=[1, 2]
        )

        ax0 = plot_image(vector_magnitude, ax=axs[0],
                         title=titles['before']['vector-fields'])
        self.overplot_vectors(ax0, v, u, num_per_dimension)

        ax1 = plot_image(vector_magnitude_centred, ax=axs[1],
                         title=titles['after']['vector-fields'] % (v_mean, u_mean))

        self.overplot_vectors(ax1, v_centred, u_centred, num_per_dimension)

        fT.remove_overlapping_tickers_for_horizontal_subplots(1, *axs)
        [fT.minor_tickers(ax) for ax in axs]
        plt.subplots_adjust(**self.fig_params)

    def show_pcc_rotation_fit(self, angles=None, recovered_angles=None,
                              poly_model_coeffs=None):
        """If phase cross correlation (pcc) was used to estimate the rotational
        offset, this method plots the fitted model and angle offsets that
        resulted in the found angle
        TODO: simplify function and make operations modular into separate functions

        Parameters
        ----------
        angles : array, optional
            The true angle offsets applied to the prealign image.
            The default is None.
        recovered_angles : array, optional
            The revoced angle offsets found when pcc'ing the rotated prealign
            image to the reference image
        poly_model_coeffs : array, optional
            The polynomial coefficients fitted to find the rotation.
            Coefficients given from highest power to lowest. The default is None.

        Raises
        ------
        ValueError
            Cannot plot the rotation model if none were supplied.
.
        Returns
        -------
        None.
        """

        if angles is None or recovered_angles is None or poly_model_coeffs is None:
            angles = self.rotation_fit['angles']
            recovered_angles = self.rotation_fit['recovered_angles']
            poly_model_coeffs = self.rotation_fit['poly_model_coeffs']

        if angles is None or recovered_angles is None or poly_model_coeffs is None:
            raise ValueError('Parameters for fitted rotation were not supplied')

        angle_diffs = angles-recovered_angles

        polynorm_func = np.poly1d(poly_model_coeffs)
        model = polynorm_func(angles)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.set_title('Angle difference')
        sc = ax.scatter(angles, angle_diffs, c=angle_diffs-model,
                        cmap=cc.m_gwv, vmin=-0.05, vmax=0.05)
        ax.plot(angles, angle_diffs, '--', color='dimgrey')

        ax.set_ylabel(r'True Angle offset - recovered offset ($^\circ$)')
        ax.set_xlabel(r'True rotation angle ($^\circ$)')

        cbar = plt.colorbar(sc, pad=0)
        cbar.set_label(r'Offset from model ($^\circ$)')

        ax.plot(angles, model, 'b-', label='y intercept=%.4f' % (poly_model_coeffs[-1]))
        plt.legend(loc='best')


def average_flow_vector(v, u, avg_func=False, **kwargs):
    """Finds the average vector of a vector field. Can choose to clip
    outliers away and/or use the median rather than the mean

    Parameters
    ----------
        v : 2d-array
            y component of a vectors field
        u : 2d-array
            x component of a vectors field
        avg_func : bool or function, optional
            Changes the average to use the median if True, or changes
            the average function to a user given function. If neither of
            these are given, the `np.nanmean` is used.
    Returns
    -------
    v_average : 2d-array
        Average y component vector
    u_average : 2d-array
        Average x component vector

    """

    if avg_func:
        avg_func = np.nanmedian

    elif not callable(avg_func):
        avg_func = np.nanmean

    v_average = avg_func(v, **kwargs)
    u_average = avg_func(u, **kwargs)

    return v_average, u_average


def sigma_clipped_mean(data, **kwargs):
    """Finds the clipped vectors of a vector field

    Parameters
    ----------
        data : nd array
            data with outliers

    Returns
    -------
    mean_clipped_data : nd-array
        Mean data ignoring outliers

    """
    mean_clipped_data = np.nanmean(basic_sigma_clip(data, **kwargs))

    return mean_clipped_data


def basic_sigma_clip(data, mask=False, **kwargs):
    """Performs basic sigma clipping with some set parameters. Mask outputs
    a boolean array of values that passed whereas no mask will return
    an array with failed data points removed

    Parameters
    ----------
    data : ndarray
        Data to be sigma clipped.
    mask : Bool, optional
        swtich for whether a mask is returned rather than a reduced
        data array. The default is False.

    Returns
    -------
    clipped_data : ndarray
        Reduced sigma clipped data. When `mask`=True, boolean array
        matching the dimensions of the input array is returned instead

    """
    sigma = kwargs.pop('sigma', 3)
    maxiters = kwargs.pop('maxiters', 3)
    mask = kwargs.pop('masked', mask)
    cenfunc = kwargs.pop('cenfunc', 'median')

    clipped_data = sigma_clip(data, sigma=sigma, maxiters=maxiters,
                              cenfunc=cenfunc, masked=mask, **kwargs)
    if mask:
        clipped_data = clipped_data.mask

    return clipped_data


def get_vector_norm(v, u):
    """Calculated the magnitude of vector components.

    Parameters
    ----------
    v : ndarray
        Vector component in the y direction.
    u : ndarray
        Vector component in the x direction.

    Returns
    -------
    norm : ndarray
        Magnitude of the given vector components with matching the
        dimensions of the input array

    """
    norm = np.sqrt(u ** 2 + v ** 2)
    return norm


def plot_vectors(v, u, y, x, ax=None):
    """Plots a vector field using the given y and x components of the vectors

    Parameters
    ----------
    v : 2darray
        Vector component in the y direction.
    u : 2darray
        Vector component in the x direction.
    x : 2darray
    y : 2darray
        For each axis, this sets the number of grid points to be calcualted.
        The default is 20
    ax : TYPE, optional
        Pre-initalised ax. The default is None.

    Returns
    -------
    None.
    """

    if ax is None:
        ax = plt.gca()

    q = ax.quiver(x, y, u, v,  **vector_plot_properties)

    _ = ax.quiverkey(q, 0.873, 0.02, 1, r'=1 pixel', labelpos='E',
                     coordinates='figure', fontproperties={'size': 'medium'})
