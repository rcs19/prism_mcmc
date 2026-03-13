import numpy as np
import pandas as pd
import corner
import emcee
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path
from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x
from main import reducedchisquared

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

if __name__ == "__main__":
    # 1. Load experimental data
    filepath_exp = Path("data/exp/98252_xrf4/sis_f3/sis_f3_no_cr.txt")
    filepath_sigma = filepath_exp.parent / (filepath_exp.stem + "_sigma" + filepath_exp.suffix)

    data_exp = pd.read_csv(filepath_exp, sep="\\s+", header=None)
    data_exp[2] = np.loadtxt(filepath_sigma, unpack=True)[1]
    data_crop = (3430,5000)
    data_exp[0]    = calibrate_x(data_exp[1], ref_eV=[3683,3934,4150], ref_idx=[143.5, 252.41, 332.65]) # 98252 t4 f3
    data_exp_cropped = data_exp[(data_exp[0]>data_crop[0]) & (data_exp[0]<data_crop[1])].reset_index(drop=True)

    xdata, ydata, ysigma = data_exp_cropped[0].values, data_exp_cropped[1].values, data_exp_cropped[2].values

    # 2c. Define parameters, initial guess, bounds and MCMC settings
    params_initial = {'tc_kev': 1.15, 'lognc': 24.3, 'ts_kev': 0.4, 'rhoR': 0.09}
    params_bounds  = {'tc_kev': (0.9, 1.3), 'lognc': (23.5, 25), 'ts_kev': (0.2, 0.5), 'rhoR': (0.08, 0.17)}
    nwalkers       = 10
    fitting_mask   = [(3450,4500)]
    directory      = "data/mcmc_run_5/"
    savefile       = "mcmc_run_5.h5"
    burn = 80       
    thin = 1
    # Initial positions of walkers
    pos = np.array([val for val in params_initial.values()]) + 0.01 * np.random.randn(nwalkers, 4) # n walkers, 4 parameters, randomise initial positions slightly
    nwalkers, ndim  = pos.shape
    sampler = emcee.backends.HDFBackend(savefile)

    # Results
    # -------
    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    labels = ["tc", "log(nc)", "ts", "rhoR"]

    #plot_chain(sampler, title="All Samples")
    plot_chain(sampler, burn=burn, thin=thin, title="Burned and thinned")

    # Corner plot
    flat_samples = sampler.get_chain(flat=True, discard=burn, thin=thin)
    fig = corner.corner(flat_samples, labels=labels,)

    # Get values which fall within 1 sigma (68% percentile)
    inds = np.arange(flat_samples.shape[0]) # sample index (for filenames)
    sigma_bounds = []
    for i in range(ndim):
        result_values = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(result_values)
        # print(f"{labels[i]} = ${result_values[1]}_{{{q[0]}}}^{{{q[1]}}}$")
        print(f"{labels[i]} = {result_values[1]:.1f} + {q[0]:.1f} - {q[1]:.1f}")

        # Samples have file format f"sample_{tc_kev:.2f}_{lognc:.2f}_{ts_kev:.2f}_{rhoR:.3f}" - obtain the bounds for these parameters from the 16 and 84 percentile values and filter the samples accordingly in the next for loop
        sigma_bounds.append((result_values[0], result_values[2]))

    directory = Path(directory)
    fig, ax = plt.subplots()
    for file in directory.rglob("*.txt"):
        file_params = file.stem.split("_")[1:-1] # extract parameters from filename
        tc_kev, lognc, ts_kev, rhoR = map(float, file_params)
        if (sigma_bounds[0][0] < tc_kev < sigma_bounds[0][1] and sigma_bounds[1][0] < lognc < sigma_bounds[1][1] and sigma_bounds[2][0] < ts_kev < sigma_bounds[2][1] and sigma_bounds[3][0] < rhoR < sigma_bounds[3][1]):
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