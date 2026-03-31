import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from src.spec import calibrate_x, get_ysigma, adjust_weights, gaussian_broadening, load_srs3p2, calibration_table, generate_gaussian_weights
from src.prism_tools import reduced_model
from main import reducedchisquared
matplotlib.rcParams.update({'font.size': 14})

folder = Path("data/exp/98252_xrf3_Mar2026/")
xdata, ydata, ysigma = load_srs3p2(folder / "sis_f1/sis_f1_no_cr.txt")
xdata = calibrate_x(ydata, ref_eV=[3420,3683,3934,4150], ref_idx=calibration_table["98252t3f1"])
# ysigma = adjust_weights(xdata, ysigma, regions=[(np.min(xdata),4000)], multiplier=0.3)
fitting_mask   = [(3580,3750), (3830,4050),]
gauss_weights = generate_gaussian_weights(xdata, centers=[3665, 3930], sigmas=[60,80], amplitude=10, baseline=0.5, n=4)
ysigma = ysigma / gauss_weights
if True:
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, label="Frame 3")
    ax.fill_between(xdata, ydata-ysigma, ydata+ysigma, color="black", alpha=0.2, label="$\\sigma$ Frame 3")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (arb.)")
    ax2 = ax.twinx()
    ax2.plot(xdata, gauss_weights, label=f"Gaussian Weights", color="red", alpha=0.5)
    ax2.set_ylabel("Weight")
    for low, high in fitting_mask:
        ax.axvspan(low, high, color="grey", alpha=0.1)

    x_min, x_max = xdata.min(), xdata.max()
    to_ps = lambda x: (x - x_min) / (x_max - x_min) * 220
    to_energy = lambda ps: ps / 220 * (x_max - x_min) + x_min
    ax2 = ax.secondary_xaxis("top", functions=(to_ps, to_energy))
    ax2.set_xlabel("Relative Time (ps)")

    ax.legend()
    plt.show()

if False:
    folder = Path("data/exp/98252_xrf4_Mar2026/")
    xdata, ydata, ysigma = load_srs3p2(folder / "sis_f3/sis_f3_no_cr.txt")
    xdata = calibrate_x(ydata, ref_eV=[3420,3683,3934,4150], ref_idx=[0, 144, 253, 332])
    fitting_mask   = [(3600,3750), (3830,4000), (4070,4500)]

    # 2c. Define parameters, initial guess, bounds and MCMC settings
    params = {'tc_kev': 1.09, 'lognc': 24.29, 'carbonmix': 0.17, 'ts_kev': 0.27, 'rhoR': 0.1}
    fitting_mask   = [(3600,3750), (3830,4000), (4070,4500)]
    corepsi        = "data/inputs/templates/core_spherical_fac_DArC.psi"
    shellpsi       = "data/inputs/templates/shell_planar_atbase_rhoR.psi"

    directory = "data/prismspect_outputs/"
    tc_kev, lognc, carbonmix, ts_kev, rhoR = params.values()
    nc = 10**lognc
    tc = tc_kev * 1e3
    ts = ts_kev * 1e3
    run_name = f"sample_{tc_kev:.2f}_{lognc:.2f}_{ts_kev:.2f}_{rhoR:.3f}"

    xmodel, ymodel = reduced_model(tc=tc, nc=nc, rc=40e-4, ts=ts, ns=20, rhoR=rhoR, carbonmix=carbonmix, 
                                   shellpsi=shellpsi,
                                   corepsi=corepsi,
                                   reuse_run=None, directory=directory, run_name=run_name, 
                                   overwrite=False, delete_prism=False, verbose=True)
    xmodel, ymodel, ymodel_bf = np.loadtxt(Path(directory) / (run_name + "_eid.txt"), unpack=True)
    ymodel = gaussian_broadening(xmodel, ymodel, R=150)
    norm_factor     = np.max(ydata)/np.max(ymodel)  
    ymodel_interp = np.interp(xdata, xmodel, ymodel) * norm_factor # scale model to data
    ymodel_bf = ymodel_bf * norm_factor

    # mask ydata and ymodel_interp 
    if fitting_mask is not None:
        mask = np.zeros_like(xdata, dtype=bool)
        for low, high in fitting_mask:
            mask |= (xdata > low) & (xdata < high)
        ydata_fit = ydata[mask]
        ymodel_interp_fit = ymodel_interp[mask]
        ysigma_fit = ysigma[mask]

    # need to find best fit amplitude first using scipy.optimize.minimize_scalar 
    res = minimize_scalar(reducedchisquared, args=(ydata_fit, ymodel_interp_fit, ysigma_fit), bounds=(0.2, 1.5), method='bounded')
    scalar = res.x

    fig, ax = plt.subplots()
    ax.plot(xdata, scalar*ymodel_interp, color="red", alpha=0.9, label="Model")
    ax.plot(xmodel, scalar*ymodel_bf, ls="--", color="red", alpha=0.9, label="Model B-F")
    ax.plot(xdata, ydata, color="black")
    ax.fill_between(xdata, ydata-ysigma, ydata+ysigma, color="gray", alpha=0.5, label="Weight")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (arb.)")
    x_min, x_max = xdata.min(), xdata.max()

    # conversion functions
    to_ps = lambda x: (x - x_min) / (x_max - x_min) * 220
    to_energy = lambda ps: ps / 220 * (x_max - x_min) + x_min

    ax2 = ax.secondary_xaxis("top", functions=(to_ps, to_energy))
    ax2.set_xlabel("Time (ps)")

    plt.show()