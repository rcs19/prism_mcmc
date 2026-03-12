import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee

np.random.seed(0)

# "True Parameters"
a = 40
p = 25
d = 1e24

# Defining test data (experimental data)
def generate_data(a, p, d, xdata, noise=True):
    ydata = a * np.sin(2*np.pi*xdata/p) - np.log10(d) 
    if noise:
        return ydata + + np.random.normal(size=len(xdata), scale=12)
    else:
        return ydata

xdata = np.linspace(0, 30, 50)
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

# Setting up MCMC
# ---------------

def log_likelihood(params, xdata, ydata, ysigma):
    a, p, d = params
    model = a * np.sin(2*np.pi*xdata/p) - d
    return -0.5 * np.sum(((ydata - model) / ysigma)**2 + np.log(2 * np.pi * ysigma**2))

def log_prior(params):
    """
    If all params are within bounds return 0.0 else return -np.inf
    This is equivalent to a prior with uniform/constant probability within the bounds and a zero probability outside (log(1)=0 and log(0)=-inf) 
    """
    a, p, d = params
    if (params_bounds['a'][0] < a < params_bounds['a'][1] and
        params_bounds['p'][0] < p < params_bounds['p'][1] and
        params_bounds['d'][0] < d < params_bounds['d'][1]):
        return 0.0
    else:
        return -np.inf 

def log_probability(params, xdata, ydata, ysigma):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xdata, ydata, ysigma)

params_initial_guess = {'a': 40, 'p': 25, 'd': 24}
params_bounds = {'a': (30, 50), 'p': (10, 40), 'd': (17, 26)}
nwalkers = 8
nsteps = 50

# Initial positions of walkers
pos = np.array([val for val in params_initial_guess.values()]) + 0.1 * np.random.randn(nwalkers, 3) # n walkers, 3 parameters
nwalkers, ndim = pos.shape

# Saving using HDF Backgend
filename = "tutorial.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata, ydata, ydata_sigma), backend=backend)
sampler.run_mcmc(pos, nsteps, progress=True)

# Results
# -------

try:
    tau = sampler.get_autocorr_time()
    print(tau)
except emcee.autocorr.AutocorrError as e:
    print(str(e))

labels = ["amp", "period", "d"]
def plot_chain(sampler, burn=0, thin=1, title=""):
    samples = sampler.get_chain(discard=burn, thin=thin)
    # Plotting the sampling chain for each walker 
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    axes[0].set_title(title)

plot_chain(sampler, title="All Samples")
plot_chain(sampler, burn=20, thin=1, title="Burned and thinned")

# Corner plot
flat_samples = sampler.get_chain(discard=20, thin=1, flat=True)
fig = corner.corner(flat_samples, labels=labels, truths=[a, p, d])
# plt.show()

fig, ax = plt.subplots()
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    ysample = sample[0] * np.sin(2*np.pi*xdata/sample[1]) - sample[2]
    ax.plot(xdata, ysample, "C1", alpha=0.1)
ax.errorbar(xdata, ydata, yerr=ydata_sigma, fmt=".k", capsize=0)
# ax.plot(x0, m_true * x0 + b_true, "k", label="truth")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Get values which fall within 1 sigma (68% percentile)
for i in range(ndim):
    result_values = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(result_values)
    # print(f"{labels[i]} = ${result_values[1]}_{{{q[0]}}}^{{{q[1]}}}$")
    print(f"{labels[i]} = {result_values[1]:.1f} + {q[0]:.1f} - {q[1]:.1f}")

plt.show()