import numpy as np
import pandas as pd
import corner
import emcee

from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar

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

# Define log-likelihood, log-prior and log-probability functions for MCMC

def reducedchisquared(a, ydata, ymodel, ysigma):
    return np.sum(((ydata - a*ymodel) / ysigma)**2) / (len(ydata)-4)

def log_likelihood(params, xdata, ydata, ysigma, fitting_mask = None, verbose=False):
    tc, lognc, ts, rhoR = params
    nc = 10**lognc

    global samplecount
    xmodel, ymodel = reduced_model(tc=tc, nc=nc, rc=40e-4, ts=ts, ns=25, rhoR=rhoR, 
    directory="data/mcmc_run1/", run_name="sample{samplecount}", 
    overwrite=False, delete_prism=True, verbose=verbose)
    ymodel = gaussian_broadening(xmodel, ymodel, R=150)
    ymodel_interp = np.interp(xdata, xmodel, ymodel) * np.max(ydata)/np.max(ymodel) # scale model to data

    # mask ydata and ymodel_interp 
    if fitting_mask is not None:
        mask = np.zeros_like(xdata, dtype=bool)
        for low, high in fitting_mask:
            mask |= (xdata > low) & (xdata < high)
        ydata = ydata[mask]
        ymodel_interp = ymodel_interp[mask]
        ysigma = ysigma[mask]

    # need to find best fit amplitude first using scipy.optimize.minimize_scalar 
    res = minimize_scalar(reducedchisquared, args=(ydata, ymodel_interp, ysigma), bounds=(0.2, 1.5), method='bounded')
    scalar = res.x
    
    return -0.5 * np.sum(((ydata - scalar*ymodel_interp) / ysigma)**2 + np.log(2 * np.pi * ysigma**2))

def log_prior(params):
    """
    If all params are within bounds return 0.0 else return -np.inf
    This is equivalent to a prior with uniform/constant probability within the bounds and a zero probability outside (log(1)=0 and log(0)=-inf) 
    """
    tc, lognc, ts, rhoR = params
    global params_bounds
    if (params_bounds['tc'][0] < tc < params_bounds['tc'][1] and
        params_bounds['lognc'][0] < lognc < params_bounds['lognc'][1] and
        params_bounds['ts'][0] < ts < params_bounds['ts'][1] and
        params_bounds['rhoR'][0] < rhoR < params_bounds['rhoR'][1]):
        return 0.0
    else:
        return -np.inf 

def log_probability(params, xdata, ydata, ysigma, fitting_mask = None, verbose=False):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xdata, ydata, ysigma, fitting_mask=fitting_mask)

# 2c. Define parameters, initial guess, bounds and MCMC settings
params_initial = {'tc': tc, 'lognc': np.log10(nc), 'ts': ts, 'rhoR': rhoR}
params_bounds  = {'tc': (900, 1300), 'lognc': (23, 25), 'ts': (200, 400), 'rhoR': (0.07, 0.14)}
nwalkers       = 10
nsteps         = 15
fitting_mask   = [(3500,3756), (3810,4250)]
verbose        = False

# Initial positions of walkers
pos = np.array([val for val in params_initial.values()]) + 0.01 * np.random.randn(nwalkers, 4) # n walkers, 4 parameters
nwalkers, ndim  = pos.shape
samplecount     = 1
sampler         = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata, ydata, ysigma, fitting_mask, verbose))
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
# plt.show()

# fig, ax = plt.subplots()
# inds = np.random.randint(len(flat_samples), size=100)
# for ind in inds:
#     sample = flat_samples[ind]
#     ysample = sample[0] * np.sin(2*np.pi*xdata/sample[1]) - sample[2]
#     ax.plot(xdata, ysample, "C1", alpha=0.1)
# ax.errorbar(xdata, ydata, yerr=ydata_sigma, fmt=".k", capsize=0)
# # ax.plot(x0, m_true * x0 + b_true, "k", label="truth")
# ax.set_xlabel("x")
# ax.set_ylabel("y")

# # Get values which fall within 1 sigma (68% percentile)
# for i in range(ndim):
#     result_values = np.percentile(flat_samples[:, i], [16, 50, 84])
#     q = np.diff(result_values)
#     # print(f"{labels[i]} = ${result_values[1]}_{{{q[0]}}}^{{{q[1]}}}$")
#     print(f"{labels[i]} = {result_values[1]:.1f} + {q[0]:.1f} - {q[1]:.1f}")

# plt.show()