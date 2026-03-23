import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from src.spec import calibrate_x, get_ysigma, adjust_weights
from main import load_srs3p2, calibration_table

matplotlib.rcParams.update({'font.size': 14})

folder = Path("data/exp/98252_xrf4_Mar2026/")
xdata, ydata, ysigma = load_srs3p2(folder / "sis_f2/sis_f2_no_cr.txt")
xdata = calibrate_x(ydata, ref_eV=[3683,3934,4150], ref_idx=calibration_table["98252t4f2"])
# ysigma = adjust_weights(xdata, ysigma, regions=[(3900,3970)], multiplier=0.5)
# ysigma = adjust_weights(xdata, ysigma, regions=[(3600,3735), (3800,np.max(xdata))], multiplier=0.5)
# ysigma = adjust_weights(xdata, ysigma, regions=[(3560,3750), (3830,3990),(4070,4400)], multiplier=0.5)

fitting_mask   = [(3550,3745), (3810,4000), (4070,4500)]

fig, ax = plt.subplots()
ax.plot(xdata, ydata, label="Frame 3")
ax.fill_between(xdata, ydata-ysigma, ydata+ysigma, color="black", alpha=0.2, label="$\\sigma$ Frame 3")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Intensity (arb.)")

for low, high in fitting_mask:
    ax.axvspan(low, high, color="grey", alpha=0.1)

x_min, x_max = xdata.min(), xdata.max()
to_ps = lambda x: (x - x_min) / (x_max - x_min) * 220
to_energy = lambda ps: ps / 220 * (x_max - x_min) + x_min
ax2 = ax.secondary_xaxis("top", functions=(to_ps, to_energy))
ax2.set_xlabel("Relative Time (ps)")

ax.legend()
plt.show()

# fig, ax = plt.subplots()
# ax.plot(np.nan, np.nan)
# # ax.set_xlim(2750, 3050)
# ax.set_xlim(3450, 5500)
# ax.set_xlabel("Energy (eV)")
# ax.set_ylabel("Intensity (arb.)")
# ax.spines["left"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.yaxis.set_visible(False)
# plt.show()
