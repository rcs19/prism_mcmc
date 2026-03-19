import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from src.spec import calibrate_x, get_ysigma

calibration_table = {
    "98252t4f2": [138., 247.0, 329.],  
    "98252t4f3": [143.5, 252, 333],  
    "98252t4f4": [134., 244.0, 329.],  
    "98263t4f2": [142.2, 250, 330],
    "98263t4f3": [142.2, 254, 339],
}

def load_srs3p2(filepath):
    # 1. Load experimental data
    filepath = Path(filepath)
    if "SRS" in filepath.stem:
        filepath_sigma = filepath.parent / (filepath.stem[:-6] + "_sigma" + filepath.suffix)
    else:
        filepath_sigma = filepath.parent / (filepath.stem + "_sigma" + filepath.suffix)
    xdata, ydata = np.loadtxt(filepath, unpack=True)
    ysigma = np.loadtxt(filepath_sigma, unpack=True)[1]
    return xdata, ydata, ysigma

folder = Path("data/exp/98263_xrf5_Mar2026/")

xabs, yabs, ysigmaabs = load_srs3p2(folder / "srs_f2_abs" / "srs_f2_abs_SRS_6_no_cr.txt")
# ysigmaabs = get_ysigma(yabs, window=100)
xfrac, yfrac, ysigmafrac = load_srs3p2(folder / "srs_f2_frac" / "srs_f2_frac_SRS_9_no_cr.txt")

fig, ax = plt.subplots()
ax.plot(xabs, ysigmaabs, color="black", label="Absolute")
ax.plot(xfrac, ysigmafrac, color="red", label="Fractional")
# ax.fill_between(xabs, yabs-ysigmaabs, yabs+ysigmaabs, color="black", alpha=0.5, label="Uncertainty")
# ax.plot(xabs, yabs, color="red", label="Absolute")
# ax.fill_between(xfrac, yfrac-ysigmafrac, yfrac+ysigmafrac, color="red", alpha=0.5, label="Fractional")
# ax.plot(xfrac, yfrac, color="blue", label="Fractional")

# ax.set_xlabel("Energy (eV)")
# ax.set_ylabel("Intensity (arb.)")

# # conversion functions
# x_min, x_max = xdata_f2.min(), xdata_f2.max()
# to_ps = lambda x: (x - x_min) / (x_max - x_min) * 220
# to_energy = lambda ps: ps / 220 * (x_max - x_min) + x_min

# ax2 = ax.secondary_xaxis("top", functions=(to_ps, to_energy))
# ax2.set_xlabel("Time (ps)")
# # ax2.set_xlim(0, 220)   # optional: force exact 0–220 range
ax.legend()
plt.show()