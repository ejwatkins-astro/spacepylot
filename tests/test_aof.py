# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:03:10 2021

@author: Liz_J
"""
__author__ = "Lis Watkins"
__copyright__ = "Liz Watkins"
__license__ = "MIT"

import importlib.resources as pkg_resources

import spacepylot.alignment as align
import spacepylot.plotting as pl

sp_path = spacepylot.__path__[0]
refpath = pkg_resources.path(data,'ref_ima.fits')
prealign_path = pkg_resources.path(data,'prealign_ima.fits')

trys = 10
shifts_op = [0,0]
rotation_op = 0

convolve_prealign = 1

op = align.AlignOpticalFlow.from_fits(prealign_path, ref_path, 
        guess_translation=shifts_op, guess_rotation=rotation_op, 
        convolve_prealign=convolve_prealign, verbose=True)
shifts_op, rotation_op = op.get_translation_rotation(niter=1)

op_plot = pl.AlignmentPlotting.from_align_object(op)

op_plot.red_blue_before_after()
op_plot.illustrate_vector_fields()


