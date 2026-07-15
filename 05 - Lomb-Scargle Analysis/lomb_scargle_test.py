##======== AIDAN'S LOMB-SCARGLE ANALYZER of DEATH [EXPERIMENTAL PROTOTYPE] ========##

from stingray.lightcurve import Lightcurve
from stingray.lombscargle import LombScarglePowerspectrum

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
from astropy.timeseries import LombScargle

rand = np.random.default_rng(42)
n = 100
t = np.sort(rand.random(n)) * 10
y = np.sin(2 * np.pi * 3.0 * t) + 0.1 * rand.standard_normal(n)
sub = np.min(y)
y -= sub
t0 = np.linspace(0, 10, 1000)
y0 = np.sin(2 * np.pi * 3.0 * t0) + 0.1 * rand.standard_normal(t0.size)
sub = np.min(y0)
y0 -= sub
spline = make_interp_spline(t, y)

lc = Lightcurve(t, y)

fig, ax = plt.subplots(1,1,figsize=(10,6))
ax.scatter(lc.time, lc.counts, lw=2, color='blue',label='lc')
ax.plot(t0, y0, lw=2, color='red',label='source of lc')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Counts (cts)")
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(which='major', width=1.5, length=7)
ax.tick_params(which='minor', width=1.5, length=4)
plt.legend()

plt.figure(1)

lps = LombScarglePowerspectrum(
    lc,
    min_freq=0,
    max_freq=None,
    method="fast",
    power_type="all",
    norm="none",
)

print(lps.freq[0:5])
print(lps.power[0:5])


fig, ax = plt.subplots(1,3,figsize=(15,6),sharey=True)
lps.plot(ax=ax[0])

ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Power")

ax[1].plot(lps.freq, lps.power.real, lw=2, color='red')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Power (Real Component)")

ax[2].plot(lps.freq, lps.power.imag, lw=2, color='blue')
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("Power (Imaginary Component)")

plt.figure(2)

##======== PERIODOGRAMS ========##

rand = np.random.default_rng(42)
t = 100 * rand.random(100)
y = np.sin(2 * np.pi * t) + 0.1 * rand.standard_normal(100)

frequency, power = LombScargle(t, y).autopower()

fig, ax = plt.subplots()
ax.plot(frequency, power)

dy = 0.1
frequency, power = LombScargle(t, y, dy).autopower()

dy = 0.1 * (1 + rand.random(100))
y = np.sin(2 * np.pi * t) + dy * rand.standard_normal(100)
frequency, power = LombScargle(t, y, dy).autopower()

plt.figure(3)

plt.show()