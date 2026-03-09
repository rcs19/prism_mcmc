import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Minimizer, create_params, report_fit, conf_interval, printfuncs
import corner

np.random.seed(0)
# Defining test data (experimental data)
def generate_data(a, p, d, xdata, noise=True):
    ydata = a * np.sin(2*np.pi*xdata/p) - np.log10(d) 
    if noise:
        return ydata + + np.random.normal(size=len(xdata), scale=12)
    else:
        return ydata

a = 40
p = 25
d = 1e24
xdata = np.linspace(0, 55, 100)
ydata = generate_data(a, p, d, xdata, noise=True)
ydata_clean = generate_data(a, p, d, xdata, noise=False)

# Standard deviation of our noisy data is obtained using the local variance of the data (around a window of 5 points)
# stddev^2 = 1/(N-1) sum((y - y_smooth)^2)
# 1. Smooth data using moving average with window size 5 points
# 2. Calculate residuals between noisy data and smoothed data (y - ysmooth)^2
# 3. Calculate sum of residuals around window sum((y - ysmooth)^2)
# 4. Divide by number of points in this window minus 1
window = 5
ydata_smooth = np.convolve(ydata, np.ones(window)/window, mode='same')
var = (ydata - ydata_smooth)**2 # residuals of each point with smoothed data
# variance of point is mean of var within window
std2 = np.convolve(var, np.ones(window), mode='same')
count = np.convolve(np.ones_like(ydata), np.ones(window), mode='same')
count = np.maximum(count,2.)
ydata_sigma = np.sqrt(std2/(count-1))

# Log-likelihood probability for MCMC `emcee`
def lnprob(params, xdata, ydata, ysigma):
    amp = params["amp"]
    period = params["period"]
    d = params["d"]
    y_model = amp * np.sin(2*np.pi*xdata/period) - d
    return -0.5 * np.sum(((y_model - ydata) / ysigma)**2 + np.log(2 * np.pi * ysigma**2))

dguess = np.log10(7e23)
dmin = np.log10(1e23)
dmax = np.log10(1e25)

params = Parameters()
params.add_many(("amp", 45, True, 30, 50),
                 ("period", 28, True, 10, 40),
                 ("d", dguess, True, dmin, dmax))
mcmc_minimizer = Minimizer(lnprob, params, fcn_args=(xdata, ydata, ydata_sigma), nan_policy='omit')
mcmc_result = mcmc_minimizer.emcee(steps=300, burn = 0, nwalkers=100, float_behavior="posterior", seed=0)

# compute the fitted model from the best-fit parameters
best_a = mcmc_result.params['amp'].value
best_p = mcmc_result.params['period'].value
best_d = mcmc_result.params['d'].value
fitted_model = best_a * np.sin(2*np.pi*xdata/best_p) - best_d

# show the minimization result
report_fit(mcmc_result)
corner_plot = corner.corner(mcmc_result.flatchain, labels=mcmc_result.var_names,
                           truths=list(mcmc_result.params.valuesdict().values()))

if True:
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = mcmc_result.flatchain
    labels = ["amp", "period", "d"]
    stepnumber = np.arange(len(samples))/100
    for i, param in enumerate(samples):
        ax = axes[i]
        ax.plot(stepnumber,samples[param], lw=0.1)
        # ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

# Plot
fig, ax = plt.subplots()
ax.plot(xdata, ydata, '+', )
ax.plot(xdata, ydata_clean, label=f"Underlying function (hidden), A={a:.1f}, P={p:.1f}", color="black", ls="--", alpha=0.9)
ax.plot(xdata, fitted_model, label=f"Maximum Likelihood, A={best_a:.1f} ± {mcmc_result.params['amp'].stderr:.2g}, P={best_p:.2f} ± {mcmc_result.params['period'].stderr:.2g}", color="red")
# ax.plot(xdata, y_models.T, color="orange", alpha=0.01)
ax.fill_between(xdata,ydata-ydata_sigma,ydata+ydata_sigma, color="black", alpha=0.1)

ax.legend()
plt.show()