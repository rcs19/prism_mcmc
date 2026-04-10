import numpy as np          
import pandas as pd         
import corner               
import emcee                

from time import time
from multiprocessing import Pool
from pathlib import Path
from matplotlib import pyplot as plt            
from scipy.optimize import minimize_scalar      

from src.spec import load_srs3p2, gaussian_broadening, calibrate_x, adjust_weights, apply_fitting_mask, generate_gaussian_weights
from src.lineratio import get_te_ne
from src.prism_tools import reduced_model      
from emcee_viewer import plot_all, plot_chain

np.random.seed(0)

# MCMC Functions : log-likelihood, log-prior and log-probability  

def reducedchisquared(a, ydata, ymodel, ysigma):
    return np.sum(((ydata - a*ymodel) / ysigma)**2) / (len(ydata)-4)

def log_likelihood(params, xdata, ydata, ysigma, corepsi, shellpsi, fitting_mask = None, reuse_run=None, directory="data/mcmc_run/",  verbose=False):
    try:
        tc_kev, lognc, carbonmix, ts_kev, rhoR = params
    except ValueError as e:
        try:
            tc_kev, lognc, ts_kev, rhoR = params
            carbonmix = None
        except ValueError as e:
            raise ValueError(f"Expected 4 or 5 parameters but got {len(params)}. Error: {str(e)}")
        
    nc = 10**lognc
    tc = tc_kev * 1e3
    ts = ts_kev * 1e3
    
    if carbonmix is not None:
        run_name = f"sample_{tc_kev:.3f}_{lognc:.2f}_{carbonmix:.2f}_{ts_kev:.2f}_{rhoR:.3f}"
    else:
        run_name = f"sample_{tc_kev:.3f}_{lognc:.2f}_{ts_kev:.2f}_{rhoR:.3f}"

    xmodel, ymodel = reduced_model(tc=tc, nc=nc, rc=40e-4, ts=ts, ns=20, rhoR=rhoR, carbonmix=carbonmix,
                                   corepsi=corepsi, shellpsi=shellpsi,
                                   reuse_run=reuse_run, directory=directory, 
                                   run_name=run_name, 
                                   overwrite=False, delete_prism=True, verbose=verbose)
    
    ymodel = gaussian_broadening(xmodel, ymodel, R=150)
    ymodel_interp = np.interp(xdata, xmodel, ymodel) * np.max(ydata)/np.max(ymodel) # scale model to data
                
    ydata_fit         = apply_fitting_mask(xdata, ydata, fitting_mask)
    ymodel_interp_fit = apply_fitting_mask(xdata, ymodel_interp, fitting_mask)
    ysigma_fit        = apply_fitting_mask(xdata, ysigma, fitting_mask)

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
    # 1. Load input deck ./data/inputs/mcmc_run_20.py
    from mcmc_saves.mcmc_run_31 import filepath, ref_eV, ref_idx, weights, params_initial, params_bounds, fitting_mask, nwalkers, nsteps, corepsi, shellpsi, directory, savefile, reuse_run, verbose

    # 2a. Load data
    xdata, ydata, ysigma = load_srs3p2(filepath)
    xdata = calibrate_x(ydata, ref_eV=ref_eV, ref_idx=ref_idx)
    if weights is not None:
        print("Using weights")
        gauss_weights = generate_gaussian_weights(xdata, centers=weights["centers"], sigmas=weights["sigmas"], amplitude=weights["amplitude"], baseline=weights["baseline"], n=weights["n"])
        ysigma = ysigma / gauss_weights

    # 2b. Initialise positions of walkers
    initial_params = np.array(list(params_initial.values()))
    pos = initial_params + 0.05 * initial_params * np.random.randn(nwalkers, len(initial_params))  # n walkers, n parameters, randomise initial positions by a fraction (e.g. 0.05) of each starting value
    nwalkers, ndim  = pos.shape

    # 2c. Initialise sampler with HDFBackend to save results to file
    backend  = emcee.backends.HDFBackend(savefile)
    backend.reset(nwalkers, ndim)

    # 3. Run MCMC sampling given inputs defined in step 1
    with Pool(processes=5) as pool:
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

    plot_chain(sampler, params=list(params_initial.keys()), burn=0, thin=1, title="All Samples")
    # plot_chain(sampler, params=labels, burn=50, thin=1, title="Burned and thinned")
    burn = 0
    plot_all(directory, sampler, burn, params_initial, xdata, ydata, ysigma, weights, fitting_mask)
    plt.show()