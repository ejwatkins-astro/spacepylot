
import spacepylot.alignment as align
import spacepylot.plotting as pl

trys = 10
shifts_op = [0,0]
rotation_op = 0

convolve_prealign = 1
ref_path = "/home/soft/python/spacepylot/src/spacepylot/data/ref_ima.fits"
prealign_path = "/home/soft/python/spacepylot/src/spacepylot/data/prealign_ima.fits"


op = align.AlignOpticalFlow.from_fits(prealign_path, ref_path, 
        guess_translation=shifts_op, guess_rotation=rotation_op, 
        convolve_prealign=convolve_prealign, verbose=True)
shifts_op, rotation_op = op.get_translation_rotation(niter=1)

op_plot = pl.AlignmentPlotting.from_align_object(op)

op_plot.red_blue_before_after()
op_plot.illustrate_vector_fields()
