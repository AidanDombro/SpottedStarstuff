##======== AIDAN'S OVERSCANNER & TRIM TESTER OF DOOM ========##

from pathlib import Path
from astropy.nddata import CCDData
import ccdproc as ccdp

OVERSCAN_SECTION = "[2099:2138, 1:2052]"
TRIM_SECTION     = "[55:2098, 1:2052]"

raw = CCDData.read("/Volumes/starstuff/Frames/ROBO_data/firstrun_calibrated/calibrated/HII 1883/20151202/20151202.292.fits", unit="adu")
print(f"Raw shape: {raw.data.shape}")           # expect (2052, 2138)

ccd = ccdp.subtract_overscan(raw, fits_section=OVERSCAN_SECTION, median=True)
ccd = ccdp.trim_image(ccd, fits_section=TRIM_SECTION)
print(f"Trimmed shape: {ccd.data.shape}")       # expect (2052, 2044)

ccd.write("/Users/aidandombrosky/Desktop/test_trimmed.fits", overwrite=True)
print("Written to /tmp/test_trimmed.fits -- please open in DS9 to verify proper trimming of frames. Godspeed!")