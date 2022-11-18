# About this package

This package has been made to automatically align astronomical images to a reference image provided the offsets are relatively small. It assumes you have already regridded and unit matched the reference to the pre-aligned image, and goes from there to provide the alignment solution.

It is still under development, and various additional methods will be added over time.

Currently, the main method being used relies on optical flow, a routine for detecting movement in images. One of the assumptions it uses are the images have the same intensity, so if one image is at lower resolution, especially the reference image, it is recommended that you match the angular resolution of your images, and mask out any oversaturated sources for the best results.

Currently, this package find both **translational offsets** and **rotational offsets**, but has an option to only search for translational offsets only. You can also provide a guess solution (or use one of the methods here to provide a good guess solution) if you want.

With the default settings, a 250x250 images takes ~5s. T

### Additional packages needed:

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

## Running the code

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

### To run, simply do:

```python
op.get_iterate_translation_rotation()
```

### Altogether:

```python
import spacepylot.alignment as align

prealign_path = 'your//path//here'
reference_path = 'your//path//here'

op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path)
op.get_iterate_translation_rotation()
```

Iterating the alignment multiple times tends to make the alignment better.

```python
op.get_iterate_translation_rotation(5)
```

### Print alignment

To look at the alignment values, we can type:

```python
print(op.translation)
print(op.rotation_deg)
```

## Guess/initial alignment

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

## Viewing the alignment visually

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

## Full script all together

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
op.get_iterate_translation_rotation(5)

#plotting
op_plot = pl.AlignmentPlotting.from_align_object(op)

op_plot.red_blue_before_after()
op_plot.before_after()
op_plot.illistrate_vector_fields()
```

## Translation only

If you want to find translation offsets only, with no rotational offsets found, we can change the homographic template used in \`op.get\_iterate\_translation_rotation\`

```python
from spacepylot.alignment_utilities import TranslationTransform

...
op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path, guess_translation=inital_guess_shifts)
op.get_iterate_translation_rotation(homography_method=TranslationTransform)
```

## Comparing solutions

If you have two solutions, to find out whih one is better, import:
```python

import spacepylot.alignment_check as ac
```

The method works by applying both solutions to the prealign image and it then runs optical flow on them compared to the reference image. The soluion with the largest offset is the worst fit. This method therefore assumes that optical flow provides the "correct" solution. If optical flow fails, you cannot use this method to quantify which solution is better.

To run the comparison run these lines for the two solutions:

```python

prealign_path = 'your//path//here'
reference_path = 'your//path//here'

mo = ac.MagnitudeOffset.from_fits(prealign_path, reference_path, solution_1, solution_2)
```

The solutions can be entered using:
- List item a homography matrix;
- List of the rotation and translation `[rotation, [x_offset, y_offset]]`

```python

prealign_path = 'your//path//here'
reference_path = 'your//path//here'

manual_offsets = [0.2 [2.8, -1.5]]

op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path)
op.get_iterate_translation_rotation(5)

mo = ac.MagnitudeOffset.from_fits(prealign_path, reference_path, manual_offsets, op.matrix_transform) 
```

If one of the solutions was just found using optical flow, you can also initalise using its object and the comparison solutions:
```python

...

op = align.AlignOpticalFlow.from_fits(prealign_path, reference_path)
op.get_iterate_translation_rotation(5)

manual_offsets = [0.2, [2.8, -1.5]]
mo = ac.MagnitudeOffset.from_op_align_object(op, manual_offsets)
```
Therefore solution 2 is the `op` solution.

To compare the two solutions, you can plot the histogram distribution of vector magnitude residuals, and you can compare the residual vector fields and their magnitude on a 2d map. 
```python
...

mo.plot_hist()
mo.plot_vector_fields()
```

To see the average and data spread (16th-84th percentiles, they are provided in a dictionary

```python
...
stats_manual_dict, stats_auto_dict = mo.get_statistics()
```
