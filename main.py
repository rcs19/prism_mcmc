import numpy as np
import pandas as pd
import corner
import emcee

from time import time
from multiprocessing import Pool
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar

from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x, adjust_weights
from src.lineratio import get_te_ne
from src.prism_tools import reduced_model

np.random.seed(0)

calibration_table = {
    "98252t4f2": [138., 247.0, 329.],  
    "98252t4f3": [143.5, 252, 333],  
    "98252t4f4": [134., 244.0, 329.],  
    "98263t4f2": [142.2, 250, 330],
    "98263t4f3": [142.2, 254, 339],
}

def load_srs3p2(filepath):
    """
    Load experimental data output by `srs3p2.pro`
    """
    # 1. Load experimental data
    filepath = Path(filepath)
    xdata, ydata = np.loadtxt(filepath, unpack=True)
    try:
        if "SRS" in filepath.stem:
            filepath_sigma = filepath.parent / (filepath.stem[:-6] + "_sigma" + filepath.suffix)
        else:
            filepath_sigma = filepath.parent / (filepath.stem + "_sigma" + filepath.suffix)
    except FileNotFoundError:
        print(f"Sigma file not found for {filepath}. Returning ysigma as ones.")
        ysigma = np.ones_like(ydata)
    ysigma = np.loadtxt(filepath_sigma, unpack=True)[1]
    return xdata, ydata, ysigma

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

# MCMC Functinos : log-likelihood, log-prior and log-probability  

def reducedchisquared(a, ydata, ymodel, ysigma):
    return np.sum(((ydata - a*ymodel) / ysigma)**2) / (len(ydata)-4)

def log_likelihood(params, xdata, ydata, ysigma, fitting_mask = None, directory="data/mcmc_run/",  verbose=False):
    tc_kev, lognc, ts_kev, rhoR = params
    nc = 10**lognc
    tc = tc_kev * 1e3
    ts = ts_kev * 1e3
    global samplecounter
    xmodel, ymodel = reduced_model(tc=tc, nc=nc, rc=40e-4, ts=ts, ns=25, rhoR=rhoR, corepsi="data/inputs/templates/core_spherical_fac.psi",
    directory=directory, run_name=f"sample_{tc_kev:.2f}_{lognc:.2f}_{ts_kev:.2f}_{rhoR:.3f}", 
    overwrite=False, delete_prism=True, verbose=verbose)
    samplecounter += 1
    ymodel = gaussian_broadening(xmodel, ymodel, R=150)
    ymodel_interp = np.interp(xdata, xmodel, ymodel) * np.max(ydata)/np.max(ymodel) # scale model to data

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
    # plt.plot(xdata, ydata, color="black", label="Data")
    # plt.plot(xdata, scalar*ymodel_interp, color="red", label="Model")
    # plt.show()
    return -0.5 * np.sum(((ydata_fit - scalar*ymodel_interp_fit) / ysigma_fit)**2 + np.log(2 * np.pi * ysigma_fit**2))

def log_prior(params):
    """
    If all params are within bounds return 0.0 else return -np.inf
    This is equivalent to a prior with uniform/constant probability within the bounds and a zero probability outside (log(1)=0 and log(0)=-inf) 
    """
    tc_kev, lognc, ts_kev, rhoR = params
    global params_bounds
    if (params_bounds['tc_kev'][0] < tc_kev < params_bounds['tc_kev'][1] and
        params_bounds['lognc'][0] < lognc < params_bounds['lognc'][1] and
        params_bounds['ts_kev'][0] < ts_kev < params_bounds['ts_kev'][1] and
        params_bounds['rhoR'][0] < rhoR < params_bounds['rhoR'][1]):
        return 0.0
    else:
        return -np.inf 

def log_probability(params, xdata, ydata, ysigma, fitting_mask = None, directory="data/mcmc_run/", verbose=False):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xdata, ydata, ysigma, fitting_mask=fitting_mask, verbose=verbose, directory=directory)

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
    samplecounter     = 1

    # Initialise sampler with HDFBackend to save results to file
    backend  = emcee.backends.HDFBackend(savefile)
    backend.reset(nwalkers, ndim)

    with Pool(processes=5) as pool:
        sampler  = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata, ydata, ysigma, fitting_mask, directory, verbose), backend=backend, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    # Results
    # -------
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    labels = ["Tc", "log(nc)", "Ts", "$\\rho$R"]

    plot_chain(sampler, params=labels, burn=0, thin=1, title="All Samples")
    plot_chain(sampler, params=labels, burn=20, thin=1, title="Burned and thinned")

    # Corner plot
    flat_samples = sampler.get_chain(flat=True)
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