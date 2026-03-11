import numpy as np
import pandas as pd
import corner
import emcee

from pathlib import Path
from matplotlib import pyplot as plt

from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x
from src.lineratio import get_te_ne
from src.prism_tools import reduced_model

# 1. Load experimental data
filepath_exp = Path("data/exp/98252_xrf4/sis_f3/sis_f3_no_cr.txt")
filepath_sigma = filepath_exp.parent / (filepath_exp.stem + "_sigma" + filepath_exp.suffix)

data_exp = pd.read_csv(filepath_exp, sep="\\s+", header=None)
data_exp[2] = np.loadtxt(filepath_sigma).T[1]
data_crop = (3430,5000)
data_exp[0]    = calibrate_x(data_exp[1], ref_eV=[3683,3934,4150], ref_idx=[143.5, 252.41, 332.65]) # 98252 t4 f3
data_exp_cropped = data_exp[(data_exp[0]>data_crop[0]) & (data_exp[0]<data_crop[1])].reset_index(drop=True)

xdata, ydata, ysigma = data_exp_cropped[0].values, data_exp_cropped[1].values, data_exp_cropped[2].values

# 2a. Lineratio Analysis for initial T_c and n_c (core electron temperature and density)
ysubtr = data_exp_cropped[1] - continuum_butterworth(data_exp_cropped[0], data_exp_cropped[1])
tc, nc = get_te_ne(data_exp_cropped[0], ysubtr, plot=False)
# 2b. Use rad-hydro for initial n_s and rhoR 
ts = 300     # eV
rho = 25     # g/cm3
rhoR = 0.09  # g/cm2

# 2c. Define parameters, initial guess, bounds and MCMC settings
params_initial = {'tc': tc, 'lognc': np.log10(nc), 'ts': ts, 'rhoR': rhoR}
params_bounds  = {'tc': (900, 1300), 'lognc': (23, 25), 'ts': (200, 400), 'rhoR': (0.07, 0.14)}
nwalkers = 10
nsteps = 100
# Initial positions of walkers
pos = np.array([val for val in params_initial.values()]) + 0.1 * np.random.randn(nwalkers, 4) # n walkers, 4 parameters
nwalkers, ndim = pos.shape

# Define log-likelihood, log-prior and log-probability functions for MCMC

def log_likelihood(params, xdata, ydata, ysigma):
    tc, lognc, ts, rhoR = params
    nc = 10**lognc
    # note!! need to do run_name = sample number
    xmodel, ymodel = reduced_model(tc=tc, nc=nc, ts=ts, ns=25, rhoR=rhoR, directory="data/mcmc_run1/", run_name="test_run", overwrite=False, delete_prism=True, verbose=False)
    ymodel_interp = np.interp(xdata, xmodel, ymodel)

    return -0.5 * np.sum(((ydata - ymodel_interp) / ysigma)**2 + np.log(2 * np.pi * ysigma**2))

def log_prior(params):
    """
    If all params are within bounds return 0.0 else return -np.inf
    This is equivalent to a prior with uniform/constant probability within the bounds and a zero probability outside (log(1)=0 and log(0)=-inf) 
    """
    tc, lognc, ts, rhoR = params
    if (params_bounds['tc'][0] < tc < params_bounds['tc'][1] and
        params_bounds['lognc'][0] < lognc < params_bounds['lognc'][1] and
        params_bounds['ts'][0] < ts < params_bounds['ts'][1] and
        params_bounds['rhoR'][0] < rhoR < params_bounds['rhoR'][1]):
        return 0.0
    else:
        return -np.inf 

def log_probability(params, xdata, ydata, ysigma):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xdata, ydata, ysigma)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata, ydata, ysigma))
sampler.run_mcmc(pos, nsteps, progress=True)