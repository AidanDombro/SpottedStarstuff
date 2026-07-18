##======== AIDAN'S VERY FUN & COOL PLACE TO TEST THINGS IN PHOTUTILS ========##

"""
https://photutils.readthedocs.io/en/stable/user_guide/index.html
"""

#=== IMPORTS ===#

import matplotlib.pyplot as plt
import numpy as np

from astropy.stats import biweight_location
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import simple_norm

from photutils.datasets import make_100gaussians_image
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
from photutils.background import Background2D, MedianBackground
from photutils.datasets import make_4gaussians_image
from photutils.centroids import (centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic, centroid_sources)
from photutils.datasets import (load_simulated_hst_star_image, make_noise_image)
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

from mpl_toolkits.axes_grid1.inset_locator import (mark_inset, zoomed_inset_axes)

#=== BACKGROUND ESTIMATION ===#

"""
# making a synthetic image of 100 sources w/ Gaussian-distributed background
# mimics any old frame of a field with multiple light sources (will apply to JM's .fits files)
# NOTE: mean of 5, standev of 2
data = make_100gaussians_image()

norm = simple_norm(data, 'sqrt', percent=99.5)
fig, ax = plt.subplots()
ax.imshow(data, norm=norm, origin='lower')

# adding a strong background gradient
ny, nx = data.shape
y, x = np.mgrid[:ny, :nx]
gradient = x * y / 5000.0
data2 = data + gradient
fig, ax = plt.subplots()
ax.imshow(data2, norm=norm, origin='lower')



sigma_clip = SigmaClip(sigma = 3.0)
bkg_estimator = MedianBackground()
bkg = Background2D(data2, (15, 15), filter_size = (3,3), sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)

print(bkg.background_median)
print(round(bkg.background_rms_median, 4))

# plotting the 2D background image
fig, ax = plt.subplots()
ax.imshow(bkg.background, origin = 'lower')

# plotting the background subtracted image
data2_sub = data2 - bkg.background
fig, ax = plt.subplots()
ax.imshow(data2_sub, norm = norm, origin = 'lower')

# image median & biweight location are both larger than background level of 5
print(np.median(data))
print(biweight_location(data))

# median absdev to estimate background noise gives value larger than 2
print(mad_std(data))

# this is called sigma clipping to remove sources from image stats
# pixels above or below a specified sigma level from the median are discarded
# better background noise level estimates--woo
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
print(np.array((mean, median, std)))

sigma_clip = SigmaClip(sigma = 3.0, maxiters = 10)
threshold = detect_threshold(data, n_sigma = 2.0, sigma_clip = sigma_clip)
segment_img = detect_sources(data, threshold, n_pixels = 10)

footprint = circular_footprint(radius = 10)
mask = segment_img.make_source_mask(footprint = footprint)
mean, emdian, std = sigma_clipped_stats(data, sigma = 3.0, mask = mask)

# plotting a mesh on top of the original image
fig, ax = plt.subplots()
ax.imshow(data, norm=norm, origin='lower')
bkg.plot_meshes(outlines=True, marker='.', color='cyan', alpha=0.3)

print(np.array((mean, median, std)))

plt.show()
"""

#=== CENTROIDS ===#

"""
data = make_4gaussians_image()
data -= np.median(data[0:30, 0:125])
data = data[40:80, 70:110]

x1, y1 = centroid_com(data)
print(np.array((x1, y1)))

x2, y2 = centroid_quadratic(data)
print(np.array((x2, y2)))

x3, y3 = centroid_1dg(data)
print(np.array((x3, y3)))

x4, y4 = centroid_2dg(data)
print(np.array((x4, y4)))

xycen1 = centroid_com(data)
xycen2 = centroid_quadratic(data)
xycen3 = centroid_1dg(data)
xycen4 = centroid_2dg(data)
xycens = [xycen1, xycen2, xycen3, xycen4]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(data, origin='lower')
marker = '+'
ms = 60
colors = ('white', 'cyan', 'red', 'blue')
labels = ('Center of Mass', 'Quadratic', '1D Gaussian', '2D Gaussian')

for xycen, color, label in zip(xycens, colors, labels):
    ax.scatter(*xycen, color=color, marker=marker, s=ms, label=label)

ax.legend(loc='lower right', fontsize=12)

ax2 = zoomed_inset_axes(ax, zoom=6, loc=9)
ax2.imshow(data, vmin=190, vmax=220, origin='lower')
ms = 1000

for xycen, color in zip(xycens, colors):
    ax2.scatter(*xycen, color=color, marker=marker, s=ms)

ax2.set_xlim(19, 21)
ax2.set_ylim(19, 21)

mark_inset(ax, ax2, loc1=3, loc2=4, fc='none', ec='black')

ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)

ax.set_xlim(0, data.shape[1] - 1)
ax.set_ylim(0, data.shape[0] - 1)

data = make_4gaussians_image()
data -= np.median(data[0:30, 0:125])
x_init = (25, 91, 151, 160)
y_init = (40, 61, 24, 71)
x, y = centroid_sources(data, x_init, y_init, box_size=25, centroid_func=centroid_2dg)

print(x)
print(y)

plt.show()
"""

#=== POINT SOURCE DETECTION ===#

"""
hdu = load_simulated_hst_star_image()
data = hdu.data + make_noise_image(hdu.data.shape, distribution='gaussian', mean=10.0, stddev=5.0, seed=0)
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

print(np.array((mean, median, std)))

hdu = load_simulated_hst_star_image()
data = hdu.data + make_noise_image(hdu.data.shape, distribution='gaussian', mean=10.0, stddev=5.0, seed=0)

mean, median, std = sigma_clipped_stats(data, sigma=3.0)
threshold = 5.0 * std

daofind = DAOStarFinder(threshold, fwhm=2.5, sharpness_range=(0.2, 1.5))
sources = daofind(data - median)

positions = np.transpose((sources['x_centroid'], sources['y_centroid']))
apertures = CircularAperture(positions, r=10.0)
norm = simple_norm(data, 'sqrt', percent=99)

fig, ax = plt.subplots()
axim = ax.imshow(data, norm=norm, origin='lower')
patches = apertures.plot(ax=ax, color='red')

plt.show()
"""

#=== APERTURE PHOTOMETRY ===#



