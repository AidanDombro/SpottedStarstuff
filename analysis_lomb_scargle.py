##======== AIDAN'S LOMB-SCARGLE ANALYZER of (MEGA)DEATH [FINISHED PRODUCT] ========##

from stingray.lightcurve import Lightcurve
from stingray.lombscargle import LombScarglePowerspectrum

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
from astropy.timeseries import LombScargle

