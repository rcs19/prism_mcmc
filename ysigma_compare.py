import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from src.spec import calibrate_x

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

folder = Path("data/exp/98252_xrf4_Mar2026/")

xdata_f2, ydata_f2, ysigma_f2 = load_srs3p2(folder / "sis_f4" / "sis_f4_no_cr.txt")
xdata_f2 = calibrate_x(ydata_f2, ref_eV=[3683,3934,4150])

# xdata_f3, ydata_f3, ysigma_f3 = load_srs3p2(folder / "sis_f3/sis_f3_no_cr.txt")
# xdata_f3 = calibrate_x(ydata_f3, ref_eV=[3683,3934,4150], ref_idx=calibration_table["98252t4f3"])

# fig, ax = plt.subplots()
# ax.plot(xdata_f2, ydata_f2, color="black",)
# ax.fill_between(xdata_f2, ydata_f2-ysigma_f2, ydata_f2+ysigma_f2, color="gray", alpha=0.5, label="Weight")

# ax.plot(xdata_f3, ydata_f3, color="red",)
# # ax.fill_between(xdata_f3, ydata_f3-ysigma_f3, ydata_f3+ysigma_f3, color="blue", alpha=0.5, label="Weight")

# ax.set_xlabel("Energy (eV)")
# ax.set_ylabel("Intensity (arb.)")

# # conversion functions
# x_min, x_max = xdata_f2.min(), xdata_f2.max()
# to_ps = lambda x: (x - x_min) / (x_max - x_min) * 220
# to_energy = lambda ps: ps / 220 * (x_max - x_min) + x_min

# ax2 = ax.secondary_xaxis("top", functions=(to_ps, to_energy))
# ax2.set_xlabel("Time (ps)")
# # ax2.set_xlim(0, 220)   # optional: force exact 0–220 range
# ax.legend()
# plt.show()