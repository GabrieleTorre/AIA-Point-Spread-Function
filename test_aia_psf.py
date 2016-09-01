from aia_calc_psf import *

psf = aia_psf( '131', npix = 901, dwavelength = 0 )

psf = psf.get()

import matplotlib.pyplot as plt
plt.imshow(psf**0.1)
plt.show()

import pdb; pdb.set_trace()
