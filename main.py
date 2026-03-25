import numpy as np          
import pandas as pd         
import corner               
import emcee                

from time import time
from multiprocessing import Pool
from pathlib import Path
from matplotlib import pyplot as plt            
from scipy.optimize import minimize_scalar      

from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x, adjust_weights, apply_fitting_mask
from src.lineratio import get_te_ne                                                                      
from src.prism_tools import reduced_model                                                                                      

np.random.seed(0)

calibration_table = {
    "98252t4f2": [138., 247.0, 329.],  
    "98252t4f3": [144, 253, 333],  
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
        ysigma = np.loadtxt(filepath_sigma, unpack=True)[1]
    except FileNotFoundError:
        print(f"Sigma file not found for {filepath}. Returning ysigma as ones.")
        ysigma = np.ones_like(ydata)
    return xdata, ydata, ysigma

def plot_chain(sampler, params, burn=0, thin=1, title=""):
    samples = sampler.get_chain(discard=burn, thin=thin)
    # Plotting the sampling chain for each walker 
    fig, axes = plt.subplots(len(params), figsize=(10, 7), sharex=True)
    for i, param in enumerate(params):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(param)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("Step Number")
    axes[0].set_title(title)
    fig.subplots_adjust(hspace=0.)

# MCMC Functions : log-likelihood, log-prior and log-probability  

def reducedchisquared(a, ydata, ymodel, ysigma):
    return np.sum(((ydata - a*ymodel) / ysigma)**2) / (len(ydata)-4)

def log_likelihood(params, xdata, ydata, ysigma, corepsi, shellpsi, fitting_mask = None, reuse_run=None, directory="data/mcmc_run/",  verbose=False):
    tc_kev, lognc, carbonmix, ts_kev, rhoR = params
    nc = 10**lognc
    tc = tc_kev * 1e3
    ts = ts_kev * 1e3
    
    xmodel, ymodel = reduced_model(tc=tc, nc=nc, rc=40e-4, ts=ts, ns=20, rhoR=rhoR, carbonmix=carbonmix,
                                   corepsi=corepsi, shellpsi=shellpsi,
                                   reuse_run=reuse_run, directory=directory, 
                                   run_name=f"sample_{tc_kev:.3f}_{lognc:.2f}_{carbonmix:.2f}_{ts_kev:.2f}_{rhoR:.3f}", 
                                   overwrite=False, delete_prism=True, verbose=verbose)
    
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
    else:
        ydata_fit = ydata
        ymodel_interp_fit = ymodel_interp
        ysigma_fit = ysigma

    # need to find best fit amplitude first using scipy.optimize.minimize_scalar 
    res = minimize_scalar(reducedchisquared, args=(ydata_fit, ymodel_interp_fit, ysigma_fit), bounds=(0.2, 1.5), method='bounded')
    scalar = res.x
    return -0.5 * np.sum(((ydata_fit - scalar*ymodel_interp_fit) / ysigma_fit)**2 + np.log(2 * np.pi * ysigma_fit**2))

def log_prior(params, params_bounds):
    """
    If all params are within bounds return 0.0 else return -np.inf
    This is equivalent to a prior with uniform/constant probability within the bounds and a zero probability outside (log(1)=0 and log(0)=-inf) 
    """
    names = list(params_bounds.keys())
    if all(params_bounds[n][0] < v < params_bounds[n][1] for v, n in zip(params, names)):
        return 0.0
    else:
        return -np.inf 

def log_probability(params, params_bounds, **likelihood_kwargs):
    lp = log_prior(params=params, params_bounds=params_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params=params, **likelihood_kwargs)

if __name__ == "__main__":
    folder = Path("data/exp/98252_xrf4_Mar2026/")
    xdata, ydata, ysigma = load_srs3p2(folder / "sis_f3/sis_f3_no_cr.txt")
    xdata = calibrate_x(ydata, ref_eV=[3420,3683,3934,4150], ref_idx=[0, 144, 253, 332])

    # 2c. Define parameters, initial guess, bounds and MCMC settings
    params_initial = {'tc_kev': 1.08, 'lognc': 24.29, 'carbonmix': 0.13, 'ts_kev': 0.3, 'rhoR': 0.1}
    params_bounds  = {'tc_kev': (0.7, 1.4), 'lognc': (23.0, 25), 'carbonmix': (0.01, 0.4), 'ts_kev': (0.1, 0.6), 'rhoR': (0.04, 0.17)}
    fitting_mask   = [(3580,3750), (3830,4050),]
    nwalkers       = 10
    nsteps         = 100
    corepsi        = "data/inputs/templates/core_spherical_atbase_leastdetailed_DArC.psi"
    shellpsi       = "data/inputs/templates/shell_planar_atbase_rhoR.psi"
    directory      = "data/mcmc_run_20/"
    savefile       = "mcmc_run_20.h5"
    reuse_run      = None 
    verbose        = True

    # Initial positions of walkers
    pos = np.array([val for val in params_initial.values()]) + 0.01 * np.random.randn(nwalkers, len(params_initial)) # n walkers, n parameters (length of parameter dict), randomise initial positions slightly
    nwalkers, ndim  = pos.shape

    # Initialise sampler with HDFBackend to save results to file
    backend  = emcee.backends.HDFBackend(savefile)
    backend.reset(nwalkers, ndim)

    with Pool(processes=10) as pool:
        sampler  = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                         args=(params_bounds,), 
                                         kwargs={"xdata": xdata, "ydata": ydata, "ysigma": ysigma, 
                                                 "fitting_mask": fitting_mask, 
                                                 "corepsi": corepsi, "shellpsi": shellpsi,  
                                                 "directory": directory, "reuse_run": reuse_run, 
                                                 "verbose": verbose},
                                         backend=backend, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    # Results #
    # ------- #
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    labels = list(params_initial.keys())

    plot_chain(sampler, params=labels, burn=0, thin=1, title="All Samples")
    # plot_chain(sampler, params=labels, burn=50, thin=1, title="Burned and thinned")

    # Corner plot
    flat_samples = sampler.get_chain(flat=True)
    corner_fig = corner.corner(flat_samples, labels=labels,)

    # Get values which fall within 1 sigma (68% percentile)
    inds = np.arange(flat_samples.shape[0]) # sample index (for filenames)
    sigma_bounds = []
    for i in range(ndim):
        result_values = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(result_values)
        print(f"{labels[i]} = {result_values[1]:.2f} + {q[0]:.2f} - {q[1]:.2f}")
        sigma_bounds.append((result_values[0], result_values[2])) # Samples have file format f"sample_{tc_kev:.2f}_{lognc:.2f}_{carbonmix:.2f}_{ts_kev:.2f}_{rhoR:.3f}" - obtain the bounds for these parameters from the 16 and 84 percentile values and filter the samples accordingly in the next for loop

    fig, ax = plt.subplots()

    for file in Path(directory).rglob("*.txt"):
        file_params = file.stem.split("_")[1:-1] # extract parameters from filename
        if all(low < val < high for val, (low, high) in zip(map(float, file_params), sigma_bounds)):
            # Load model spectrum
            xmodel, ymodel       = np.loadtxt(file, usecols=(0, 1), unpack=True)
            try:
                ymodel_bf, ymodel_ff = np.loadtxt(file, usecols=(2, 3), unpack=True)
            except ValueError:
                ymodel_bf, ymodel_ff = np.zeros_like(ymodel), np.zeros_like(ymodel)

            # Broaden according to instrument spectral resolving power R 
            ymodel, ymodel_bf, ymodel_ff = gaussian_broadening(xmodel, ymodel, R=150), gaussian_broadening(xmodel, ymodel_bf, R=150), gaussian_broadening(xmodel, ymodel_ff, R=150)

            # Must normalise to experimental scale before obtaining best fit amplitude using minimize_scalar (otherwise it doesn't work)
            norm_factor     = np.max(ydata)/np.max(ymodel)
            ymodel_interp   = np.interp(xdata, xmodel, ymodel) * norm_factor
            ymodel_bf, ymodel_ff = ymodel_bf*norm_factor, ymodel_ff*norm_factor
                
            ydata_fit         = apply_fitting_mask(xdata, ydata, fitting_mask)
            ymodel_interp_fit = apply_fitting_mask(xdata, ymodel_interp, fitting_mask)
            ysigma_fit        = apply_fitting_mask(xdata, ysigma, fitting_mask)

            # need to find best fit amplitude first using scipy.optimize.minimize_scalar 
            res = minimize_scalar(reducedchisquared, args=(ydata_fit, ymodel_interp_fit, ysigma_fit), bounds=(0.2, 1.5), method='bounded')
            scalar = res.x

            # add to plot
            ax.plot(xdata, scalar*ymodel_interp, color="red", alpha=0.05)
            ax.plot(xmodel, scalar*ymodel_bf, ls="--", color="red", alpha=0.05,)
            ax.plot(xmodel, scalar*ymodel_ff, ls=":", color="red", alpha=0.05,)
    
    if fitting_mask is not None:
        for low, high in fitting_mask:
            ax.axvspan(low, high, color="grey", alpha=0.1)
            
    ax.plot(xdata, ydata, color="black", label="Data")
    ax.fill_between(xdata, ydata-ysigma, ydata+ysigma, color="gray", alpha=0.5, label="Weight")

    # Dummy plots for label
    ax.plot(np.nan, np.nan, color="red", alpha=0.5, label="Model")
    ax.plot(np.nan, np.nan, ls="--", color="red", alpha=0.5, label="Model BF")
    ax.plot(np.nan, np.nan, ls=":", color="red", alpha=0.05, label="Model FF")
    ax.axvspan(np.nan, np.nan, color="grey", alpha=0.1, label="Fitting Mask")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (arb.)")

    # Top time axis (conversion functions)
    x_min, x_max = xdata.min(), xdata.max()
    to_ps = lambda x: (x - x_min) / (x_max - x_min) * 220
    to_energy = lambda ps: ps / 220 * (x_max - x_min) + x_min
    ax2 = ax.secondary_xaxis("top", functions=(to_ps, to_energy))
    ax2.set_xlabel("Time (ps)")

    plt.show()