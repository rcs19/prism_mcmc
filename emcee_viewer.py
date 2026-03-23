import numpy as np
import pandas as pd
import corner
import emcee
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path
from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x
from main import reducedchisquared, load_srs3p2, calibration_table

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
    ysigma = adjust_weights(xdata, ysigma, regions=[(3900,3970)], multiplier=0.5)
    ysigma = adjust_weights(xdata, ysigma, regions=[(3600,3735), (3800,np.max(xdata))], multiplier=0.5)
    ysigma = adjust_weights(xdata, ysigma, regions=[(3560,3750), (3830,3990),(4070,4400)], multiplier=0.5)

    # 2c. Define parameters, initial guess, bounds and MCMC settings
    params_initial = {'tc_kev': 1.10, 'lognc': 24.3, 'ts_kev': 0.4, 'rhoR': 0.10}
    params_bounds  = {'tc_kev': (0.9, 1.4), 'lognc': (23.5, 25), 'ts_kev': (0.2, 0.55), 'rhoR': (0.08, 0.17)}
    nwalkers       = 10
    nsteps         = 110
    fitting_mask   = [(3550,3745), (3810,4000), (4070,4500)]
    verbose        = True
    directory      = "data/mcmc_run_10/"
    savefile = "mcmc_run_10.h5"

    # Initial positions of walkers
    pos = np.array([val for val in params_initial.values()]) + 0.01 * np.random.randn(nwalkers, 4) # n walkers, 4 parameters, randomise initial positions slightly
    nwalkers, ndim  = pos.shape
    sampler = emcee.backends.HDFBackend(savefile)

    # Results
    # -------
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    labels = ["Tc", "log(nc)", "Ts", "$\\rho$R"]

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
        # print(f"{labels[i]} = ${result_values[1]}_{{{q[0]}}}^{{{q[1]}}}$")
        print(f"{labels[i]} = {result_values[1]:.2f} + {q[0]:.2f} - {q[1]:.2f}")

        # Samples have file format f"sample_{tc_kev:.2f}_{lognc:.2f}_{ts_kev:.2f}_{rhoR:.3f}" - obtain the bounds for these parameters from the 16 and 84 percentile values and filter the samples accordingly in the next for loop
        sigma_bounds.append((result_values[0], result_values[2]))

    directory = Path(directory)
    fig, ax = plt.subplots()

    for file in directory.rglob("*.txt"):
        file_params = file.stem.split("_")[1:-1] # extract parameters from filename
        tc_kev, lognc, ts_kev, rhoR = map(float, file_params)
        if (sigma_bounds[0][0] < tc_kev < sigma_bounds[0][1] and sigma_bounds[1][0] < lognc < sigma_bounds[1][1] and sigma_bounds[2][0] < ts_kev < sigma_bounds[2][1] and sigma_bounds[3][0] < rhoR < sigma_bounds[3][1]):
            xmodel, ymodel = np.loadtxt(file, unpack=True)
            # mask ydata and ymodel_interp 
            ymodel = gaussian_broadening(xmodel, ymodel, R=150)
            ymodel_interp = np.interp(xdata, xmodel, ymodel) * np.max(ydata)/np.max(ymodel)

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
            ax.plot(xdata, scalar*ymodel_interp, color="red", alpha=0.05)
    
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