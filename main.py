import numpy as np
import pandas as pd
import corner
import emcee

from time import time
from multiprocessing import Pool
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar

from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x
from src.lineratio import get_te_ne
from src.prism_tools import reduced_model

# Define log-likelihood, log-prior and log-probability functions for MCMC

def reducedchisquared(a, ydata, ymodel, ysigma):
    return np.sum(((ydata - a*ymodel) / ysigma)**2) / (len(ydata)-4)

def log_likelihood(params, xdata, ydata, ysigma, fitting_mask = None, directory="data/mcmc_run/",  verbose=False):
    tc_kev, lognc, ts_kev, rhoR = params
    nc = 10**lognc
    tc = tc_kev * 1e3
    ts = ts_kev * 1e3
    global samplecounter
    xmodel, ymodel = reduced_model(tc=tc, nc=nc, rc=40e-4, ts=ts, ns=25, rhoR=rhoR, 
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
    filepath_exp = Path("data/exp/98252_xrf4/sis_f3/sis_f3_no_cr.txt")
    filepath_sigma = filepath_exp.parent / (filepath_exp.stem + "_sigma" + filepath_exp.suffix)

    data_exp = pd.read_csv(filepath_exp, sep="\\s+", header=None)
    data_exp[2] = np.loadtxt(filepath_sigma).T[1]
    data_crop = (3430,5000)
    data_exp[0]    = calibrate_x(data_exp[1], ref_eV=[3683,3934,4150], ref_idx=[143.5, 252.41, 332.65]) # 98252 t4 f3
    data_exp_cropped = data_exp[(data_exp[0]>data_crop[0]) & (data_exp[0]<data_crop[1])].reset_index(drop=True)

    xdata, ydata, ysigma = data_exp_cropped[0].values, data_exp_cropped[1].values, data_exp_cropped[2].values

    # 2c. Define parameters, initial guess, bounds and MCMC settings
    params_initial = {'tc_kev': 1.15, 'lognc': 24.3, 'ts_kev': 0.4, 'rhoR': 0.09}
    params_bounds  = {'tc_kev': (0.9, 1.3), 'lognc': (23.5, 25), 'ts_kev': (0.2, 0.5), 'rhoR': (0.07, 0.14)}
    nwalkers       = 10
    nsteps         = 150
    fitting_mask   = [(3500,3756), (3810,4400)]
    verbose        = True
    directory      = "data/mcmc_run_kev_multi/"
    # Initial positions of walkers
    pos = np.array([val for val in params_initial.values()]) + 0.01 * np.random.randn(nwalkers, 4) # n walkers, 4 parameters
    nwalkers, ndim  = pos.shape
    samplecounter     = 1

    # Initialise sampler with HDFBackend to save results to file
    filename = "mcmc_run.h5"
    backend  = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    with Pool(processes=10) as pool:
        sampler  = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata, ydata, ysigma, fitting_mask, directory, verbose), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    # Results
    # -------
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    labels = ["tc", "log(nc)", "ts", "rhoR"]
    def plot_chain(sampler, burn=0, thin=1, title=""):
        samples = sampler.get_chain(discard=burn, thin=thin)
        # Plotting the sampling chain for each walker 
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        axes[0].set_title(title)
        fig.subplots_adjust(hspace=0.)

    plot_chain(sampler, title="All Samples")
    # plot_chain(sampler, burn=20, thin=1, title="Burned and thinned")

    # Corner plot
    flat_samples = sampler.get_chain(flat=True)
    fig = corner.corner(flat_samples, labels=labels,)
    plt.show()

    # Get values which fall within 1 sigma (68% percentile)
    inds = np.arange(flat_samples.shape[0]) # sample index (for filenames)
    for i in range(ndim):
        result_values = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(result_values)
        # print(f"{labels[i]} = ${result_values[1]}_{{{q[0]}}}^{{{q[1]}}}$")
        print(f"{labels[i]} = {result_values[1]:.1f} + {q[0]:.1f} - {q[1]:.1f}")

        # filter flat_samples to only include those within 1 sigma of the median for each parameter
        #mask = (flat_samples[:, i] > result_values[0]) & (flat_samples[:, i] < result_values[2])
        #flat_samples = flat_samples[mask]
        #inds = inds[mask]

    directory = Path(directory)
    fig, ax = plt.subplots()
    for file in directory.rglob("*.txt"):
        #sampleparams = flat_samples[ind]
        #sampledata = np.loadtxt(f"{directory}sample{ind+1}_eid.txt").T
        sampledata = np.loadtxt(file).T
        xmodel, ymodel = sampledata[0], sampledata[1]
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
    ax.errorbar(xdata, ydata, yerr=ysigma, color="black", capsize=0)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (arb.)")
    plt.show()