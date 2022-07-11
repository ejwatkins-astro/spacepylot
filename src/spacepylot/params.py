# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:44:01 2021

@author: Liz_J
"""
__author__ = "Elizabeth Watkins, Eric Emsellem"
__copyright__ = "Elizabeth Watkins"
__license__ = "MIT License"
__contact__ = "<liz@email"

from skimage.filters import difference_of_gaussians

pcc_params = {
    'histogram_equalisation': True,
    'remove_boundary_pixels': 25,
    'hpf': difference_of_gaussians,
    'hpf_kwargs': {
        'low_sigma': 0.9,
        'high_sigma': 0.9 * 8
    },
}
