---
title: astroAlign-readme
updated: 2022-07-11 13:56:46Z
created: 2022-03-15 15:38:00Z
latitude: 49.41240000
longitude: 8.69490000
altitude: 0.0000
---

# About this package

This package has been made to automatically align astronomical images to a reference image provided the offsets are relatively small. It assumes you have already regridded and unit matched the reference to the pre-aligned image, and goes from there to provide the alignment solution.

It is still under development, and various additional methods will be added over time.

Currently, the main method being used relies on optical flow, a routine for detecting movement in images. One of the assumptions it uses are the images have the same intensity, so if one image is at lower resolution, especially the reference image, it is recommended that you match the angular resolution of your images, and mask out any oversaturated sources for the best results.

Currently, this package find both **translational offsets** and **rotational offsets**, but has an option to only search for translational offsets only. You can also provide a guess solution (or use one of the methods here to provide a good guess solution) if you want.

With the current settings, a 250x250 images takes ~5-10s. These settings resulting in this speed cannot currently be accessed by the user but a future update will.

Additional packages needed:

`numpy`

`matplotlib`

`scipy`

`astropy`

`reproject`

`skimage`

`colorcet`

* * *

# Getting started

To use this package first import it:

```python
import spaceplot.alignment as align
```

To accurately find translation and rotational offsets use:

```python
align.AlignOpticalFlow
```

To initialise the routine, we can either manually give the object the realign and reference images:

```python
op = align.AlignOpticalFlow(prealign_image, reference_image)
```

or we can load them straight from file if they are saved in the fits format:

```python
op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path)
```

To run, simply do:

```python
shifts_op, rotation_op = op.get_iterate_translation_rotation()
```

Altogether:

```python
import spacepylot.alignment as align

prealign_path = 'your//path//here'
reference_path = 'your//path//here'

op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path)
shifts_op, rotation_op = op.get_iterate_translation_rotation()
```

Iterating the alignment multiple times tends to make the alignment better. 

```python
shifts_op, rotation_op = op.get_iterate_translation_rotation(5)
```

If you want to provide an initial guess, add that guess when initialised:

```python
op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path, guess_translation=[2,4], guess_rotation=0.1)
```

Offsets are given in pixels, and rotational offsets are given in degrees.

For larger offsets (>5-10 pixels), this routine might struggle to find the solution. For these cases, you can find a good initial guess for the translation using phase cross correlation:

```python
gt = align.AlignTranslationPCC.from_fits(prealign_path, reference_path)
inital_guess_shifts = gt.get_translation(split_image=2)

op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path, guess_translation=inital_guess_shifts)
```

`split_image=2`  splits the analysis 2x2=4 times to help provide a more robust solution.

If one of the images are at a higher resolution, you can convolve them when initialising the alignment object as a quick way of helping the alignment. Units of convolve are in pixels.

```python
op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path, convolve_prealign=1)
```

Finally to inspect your solution, import the plotting package,

```python
import spacepylot.plotting as pl
```

and initialise the plotting using the align object:

```python
op_plot = pl.AlignmentPlotting.from_align_object(op)
```

To view the differences before and after, you can look at them using red-blue overlap, or a simple subtraction:

```python
op_plot.red_blue_before_after()
op_plot.before_after()
```

We can also look at the vectors that were found before, and after removing the mean translation (not rotation). This helps demonstrate if a rotation offset is real as the vectors with curl around. The colourmap then indicates the vector magnitude

```python
op_plot.illistrate_vector_fields()
```

Altogether:

```python
import spacepylot.alignment as align
import spacepylot.plotting as pl

prealign_path = 'your//path//here'
reference_path = 'your//path//here'

#get an inital guess of the translation
gt = align.AlignTranslationPCC.from_fits(prealign_path, reference_path)
inital_guess_shifts = gt.get_translation(split_image=2)

op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path, guess_translation=inital_guess_shifts)
#solution
shifts_op, rotation_op = op.get_iterate_translation_rotation(5)

#plotting
op_plot = pl.AlignmentPlotting.from_align_object(op)

op_plot.red_blue_before_after()
op_plot.before_after()
op_plot.illistrate_vector_fields()
```

Finally, if you want to find translation offsets only, with no rotational offsets found, we can change the homographic template used in \`op.get\_translation\_rotation\`

```python
import spacepylot.translational_transform as tt

...
op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path, guess_translation=inital_guess_shifts)
shifts_op, rotation_op = op.get_translation_rotation(homography_method=tt.TranslationTransform)
```