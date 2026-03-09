import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Minimizer, create_params, report_fit, conf_interval, printfuncs
import corner

# Defining test data (experimental data)
np.random.seed(0)
a = 40
p = 25
xdata = np.linspace(0, 55, 100)
ydata = a * np.sin(2*np.pi*xdata/p) + np.random.normal(size=len(xdata), scale=12)
ydata_clean = a * np.sin(2*np.pi*xdata/p)

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

# Objective function (residuals)
def objective(params, xdata, ydata, ydata_sigma):
    amp = params["amp"]
    period = params["period"]
    y_model = amp * np.sin(2*np.pi*xdata/period)
    return (ydata - y_model) / ydata_sigma

# Log-prior probability for MCMC `emcee`
def lnprob(params, xdata, ydata, ysigma):
    amp = params["amp"]
    period = params["period"]
    y_model = amp * np.sin(2*np.pi*xdata/period)
    return -0.5 * np.sum(((y_model - ydata) / ysigma)**2 + np.log(2 * np.pi * ysigma**2))

params = Parameters()
params.add_many(("amp", 45, True, 30, 50),
                 ("period", 28, True, 10, 40))
mcmc_minimizer = Minimizer(lnprob, params, fcn_args=(xdata, ydata, ydata_sigma), nan_policy='omit')
mcmc_result = mcmc_minimizer.emcee(steps=300, burn = 100, nwalkers=100, float_behavior="posterior", seed=0)

# compute the fitted model from the best-fit parameters
best_a = mcmc_result.params['amp'].value
best_p = mcmc_result.params['period'].value
fitted_model = best_a * np.sin(2*np.pi*xdata/best_p)

# show the minimization result
report_fit(mcmc_result)
emcee_plot = corner.corner(mcmc_result.flatchain, labels=mcmc_result.var_names,
                           truths=list(mcmc_result.params.valuesdict().values()))

# Plot
fig, ax = plt.subplots()
y_model = best_a * np.sin(2*np.pi*xdata/best_p)
ax.plot(xdata, ydata, '+', )
ax.plot(xdata, ydata_clean, label=f"Underlying function (hidden), A={a:.1f}, P={p:.1f}", color="black", ls="--", alpha=0.9)
ax.plot(xdata, y_model, label=f"Maximum Likelihood, A={best_a:.1f} ± {mcmc_result.params['amp'].stderr:.2g}, P={best_p:.2f} ± {mcmc_result.params['period'].stderr:.2g}", color="red")
# ax.plot(xdata, y_models.T, color="orange", alpha=0.01)
ax.fill_between(xdata,ydata-ydata_sigma,ydata+ydata_sigma, color="black", alpha=0.1)

ax.legend()
plt.show()