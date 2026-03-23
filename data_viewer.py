import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from src.spec import calibrate_x, get_ysigma, adjust_weights

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

def plot_spec(xdata, ydata, ysigma = None, ax = None, label=""):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(xdata, ydata, color="black", label=label)
    if ysigma is not None:
        ax.fill_between(xdata, ydata-ysigma, ydata+ysigma, color="black", alpha=0.2, label=f"$\\sigma$ {label}")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (arb.)")
    ax.legend()
    return ax

folder = Path("data/exp/98252_xrf4_Mar2026/")
xdata, ydata, ysigma = load_srs3p2(folder / "sis_f3/sis_f3_no_cr.txt")
ysigma = adjust_weights(xdata, ysigma, regions=[(3560,3750), (3830,3990),(4070,4400)], multiplier=0.2)

fig, ax = plt.subplots()
plot_spec(xdata, ydata, ysigma, ax=ax)

x_min, x_max = xdata.min(), xdata.max()
to_ps = lambda x: (x - x_min) / (x_max - x_min) * 220
to_energy = lambda ps: ps / 220 * (x_max - x_min) + x_min
ax2 = ax.secondary_xaxis("top", functions=(to_ps, to_energy))
ax2.set_xlabel("Time (ps)")

ax.legend()
plt.show()