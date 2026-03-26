import numpy as np      
import pandas as pd     
import corner           
import emcee            
from matplotlib import pyplot as plt    
from scipy.optimize import minimize_scalar  
from pathlib import Path
from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x, adjust_weights, apply_fitting_mask
from main import plot_chain, reducedchisquared, load_srs3p2, calibration_table

def plot_all(directory, sampler, burn, params_initial, xdata, ydata, ysigma, weights, fitting_mask):
    """
    Given experimental data, MCMC sampler results, and model outputs, plot:
    1. Corner plot of MCMC samples after discarding n=burn samples
    2. Overlay of model outputs which fall within 1 sigma of the MCMC samples on top of experimental data
    """
    directory = Path(directory)
    labels = list(params_initial.keys())
    ndim = len(labels)

    # Corner plot
    flat_samples = sampler.get_chain(flat=True, discard=burn, thin=1)
    fig_corner = corner.corner(flat_samples, labels=labels,)

    # Get values which fall within 1 sigma (68% percentile)
    sigma_bounds = []
    for i in range(ndim):
        result_values = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(result_values)
        print(f"{labels[i]} = {result_values[1]:.2f} + {q[0]:.2f} - {q[1]:.2f}")
        sigma_bounds.append((result_values[0], result_values[2]))

    fig, ax = plt.subplots()

    for file in directory.rglob("*.txt"):
        file_params = file.stem.split("_")[1:-1] # extract parameters from filename
        # this if statement filters for parameters which fall within 1 sigma of parameter results
        if all(low < val < high for val, (low, high) in zip(map(float, file_params), sigma_bounds)):
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

if __name__ == "__main__":

    # 1. Load input deck ./data/inputs/mcmc_run_20.py
    from data.inputs.mcmc_run_20 import filepath, ref_eV, ref_idx, weights, params_initial, params_bounds, fitting_mask, nwalkers, nsteps, corepsi, shellpsi, directory, savefile, reuse_run, verbose

    # 2a. Load data
    xdata, ydata, ysigma = load_srs3p2(filepath)
    xdata = calibrate_x(ydata, ref_eV=ref_eV, ref_idx=ref_idx)
    
    if weights is not None:
        print("Using weights")

    sampler = emcee.backends.HDFBackend(savefile)

    # Results #
    # ------- #
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    labels = list(params_initial.keys())
    
    plot_chain(sampler, params=labels, burn=0, thin=1, title="All Samples")
    burn = 50
    plot_chain(sampler, params=labels, burn=burn, thin=1, title=f"First {burn} samples discarded")
