##======== AIDAN'S LOMB-SCARGLE ANALYZER of (MEGA)DEATH ========##

from stingray.lightcurve import Lightcurve
from stingray.lombscargle import LombScarglePowerspectrum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
from astropy.timeseries import LombScargle

df = pd.read_excel("/Volumes/starstuff/Frames/NURO_2011/plots/hii_1883_cleanish.xlsx")

print("Columns found in your file:")
print(list(df.columns))

x = df['HJD']
y = df['DiffMag']

plt.figure(figsize=(10,6))
plt.scatter(x, y, alpha = 0.5, edgecolors = 'w')

plt.title('Scatter HJD-DiffMag Test')
plt.xlabel('Heliocentric Julian Date')
plt.ylabel('Differential Magnitude')

plt.show()
