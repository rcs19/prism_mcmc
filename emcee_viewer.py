import numpy as np      # type: ignore
import pandas as pd     # type: ignore
import corner           # type: ignore
import emcee            # type: ignore
from matplotlib import pyplot as plt    # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
from pathlib import Path
from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x, adjust_weights        # type: ignore
from main import reducedchisquared, load_srs3p2, calibration_table  # type: ignore

def plot_chain(sampler, params, burn=0, thin=1, title=""):
    samples = sampler.get_chain(discard=burn, thin=thin)
    # Plotting the sampling chain for each walker 
    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    for i, param in enumerate(params):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(param)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("Step Number")
    axes[0].set_title(title)
    fig.subplots_adjust(hspace=0.)

if __name__ == "__main__":
    # 1. Load experimental data
    folder = Path("data/exp/98252_xrf4_Mar2026/")
    xdata, ydata, ysigma = load_srs3p2(folder / "sis_f3/sis_f3_no_cr.txt")
    xdata = calibrate_x(ydata, ref_eV=[3683,3935,4150], ref_idx=calibration_table["98252t4f3"])

    # 2c. Define parameters, initial guess, bounds and MCMC settings
    params_initial = {'tc_kev': 1.0, 'lognc': 24.46, 'ts_kev': 0.5, 'rhoR': 0.08}
    params_bounds  = {'tc_kev': (0.7, 1.4), 'lognc': (23.0, 25), 'ts_kev': (0.2, 0.55), 'rhoR': (0.05, 0.17)}
    fitting_mask   = [(3450,3745), (3810,4020), (4070,4400)]
    nwalkers       = 10
    nsteps         = 120
    directory      = "data/mcmc_run_15/"
    savefile       = "mcmc_run_15.h5"
    reuse_run      = None 
    verbose        = True

    # Initial positions of walkers
    pos = np.array([val for val in params_initial.values()]) + 0.01 * np.random.randn(nwalkers, len(params_initial)) # n walkers, n parameters (length of parameter dict), randomise initial positions slightly
    nwalkers, ndim  = pos.shape    
    sampler = emcee.backends.HDFBackend(savefile)

    # Results
    # -------
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    labels = list(params_initial.keys())

    plot_chain(sampler, params=labels, burn=0, thin=1, title="All Samples")
    plot_chain(sampler, params=labels, burn=80, thin=1, title="First 80 Samples Discarded")

    # Corner plot
    flat_samples = sampler.get_chain(flat=True, discard=80, thin=1)
    fig = corner.corner(flat_samples, labels=labels,)

    # Get values which fall within 1 sigma (68% percentile)
    inds = np.arange(flat_samples.shape[0]) # sample index (for filenames)
    sigma_bounds = []
    for i in range(ndim):
        result_values = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(result_values)
        print(f"{labels[i]} = {result_values[1]:.2f} + {q[0]:.2f} - {q[1]:.2f}")
        sigma_bounds.append((result_values[0], result_values[2]))

    directory = Path(directory)
    fig, ax = plt.subplots()

    for file in directory.rglob("*.txt"):
        file_params = file.stem.split("_")[1:-1] # extract parameters from filename
        tc_kev, lognc, ts_kev, rhoR = map(float, file_params)
        if all(low < val < high for val, (low, high) in zip(map(float, file_params), sigma_bounds)):
            xmodel, ymodel, ymodel_bf = np.loadtxt(file, unpack=True)
            # mask ydata and ymodel_interp 
            ymodel = gaussian_broadening(xmodel, ymodel, R=150)
            ymodel_bf = gaussian_broadening(xmodel, ymodel_bf, R=150)
            norm_factor     = np.max(ydata)/np.max(ymodel)
            ymodel_interp   = np.interp(xdata, xmodel, ymodel) * norm_factor
            ymodel_bf       = ymodel_bf * norm_factor

            if fitting_mask is not None:
                mask = np.zeros_like(xdata, dtype=bool)
                for low, high in fitting_mask:
                    mask |= (xdata > low) & (xdata < high)
                ydata_fit = ydata[mask]
                ymodel_interp_fit = ymodel_interp[mask]
                ysigma_fit = ysigma[mask]
            else:
                ydata_fit = ydata
                ymodel_interp_fit = ymodel_interp
                ysigma_fit = ysigma

            # need to find best fit amplitude first using scipy.optimize.minimize_scalar 
            res = minimize_scalar(reducedchisquared, args=(ydata_fit, ymodel_interp_fit, ysigma_fit), bounds=(0.2, 1.5), method='bounded')
            scalar = res.x
            ax.plot(xdata, scalar*ymodel_interp, color="red", alpha=0.05)
            ax.plot(xmodel, scalar*ymodel_bf, ls="--", color="red", alpha=0.05, label="Model B-F")

    if fitting_mask is not None:
        for low, high in fitting_mask:
            ax.axvspan(low, high, color="grey", alpha=0.1)
            
    ax.plot(xdata, ydata, color="black")
    ax.fill_between(xdata, ydata-ysigma, ydata+ysigma, color="gray", alpha=0.5, label="Weight")
    ax.plot(np.nan, np.nan, color="red", alpha=0.2, label="Model")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (arb.)")
    x_min, x_max = xdata.min(), xdata.max()

    # conversion functions
    to_ps = lambda x: (x - x_min) / (x_max - x_min) * 220
    to_energy = lambda ps: ps / 220 * (x_max - x_min) + x_min

    ax2 = ax.secondary_xaxis("top", functions=(to_ps, to_energy))
    ax2.set_xlabel("Time (ps)")

    plt.show()