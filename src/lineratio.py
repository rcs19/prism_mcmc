import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def fitfunc_two_gaussians(x, *params):
    """
    Double gaussian fitting function for scipy.curve_fit
    """
    a1, c1, s1, a2, c2, s2, k = params
    gaussian1 = a1 * np.exp(-((x - c1) / (np.sqrt(2) * s1))**2)
    gaussian2 = a2 * np.exp(-((x - c2) / (np.sqrt(2) * s2))**2)
    return gaussian1 + gaussian2 + k

def get_fwhm_and_area(*popt, pcov, R = None, verbose=False):
    """
    Assumes parameters are [amplitude, centre, stddev] for a single gaussian.
    Returns FWHM and area under the gaussian. Optionally specify spectral resolution R to deconvolve FWHM. This then returns the true linewidth.

    Arguments
    ---------
    popt : array
        Optimal values for the parameters.
    pcov : 2-D array
        The estimated covariance of popt.
    R : float, optional
        Spectral resolution E/dE. If provided, FWHM and gaussian is deconvolved accordingly (reverses effect of instrumental broadening).

    Returns
    -------
    fwhm : float
        Full-width at half-maximum of the gaussian (deconvolved if R provided).
    fwhm_err : float
        Uncertainty in FWHM.
    area : float
        Area under the gaussian.
    area_err : float
        Uncertainty in area.
    """
    # Amplitude, centre, standard deviation
    a, c, s = popt  
    a_err, c_err, s_err = np.sqrt(np.diag(pcov))
    
    fwhm = 2 * np.sqrt(2 * np.log(2)) * s
    fwhm_err = 2 * np.sqrt(2 * np.log(2)) * s_err

    if R is not None:
        fwhm_inst = c / R
        lw = np.sqrt(fwhm**2 - fwhm_inst**2) # Linewidth
        # Simple error propagation assuming no error on fwhm_inst
        lw_err = (fwhm / lw) * fwhm_err #  analytical error: lw_err^2 = d(lw)/d(FWHM)*FWHM_err

        # Reassign deconvolved values
        fwhm = lw
        fwhm_err = lw_err
        s = fwhm / (2 * np.sqrt(2 * np.log(2))) # New standard deviation after deconvolution

    area = a * s * np.sqrt(2 * np.pi)
    area_err = np.sqrt((s * np.sqrt(2 * np.pi) * a_err)**2 + (a * np.sqrt(2 * np.pi) * s_err)**2)

    if verbose:
        print(f"FWHM = {fwhm:.2f} ± {fwhm_err:.2f} eV, Area = {area:.2f} ± {area_err:.2f}")

    return fwhm, fwhm_err, area, area_err

def get_te_ne(xdata, ydata, fitting_region=(3450, 4100), sat_masks=[(3550,3660), (3790,3905)], plot=False, round=True):
    """
    Fit two Gaussians centred on He-beta (3680 eV) and Ly-beta (3935 eV) lines, masking satellite regions. 
    Returns electron temperature (assuming n_e = 1e24 cm-3) and density (assuming T_e = 1000 eV) from lookup tables (Keane 1993, Gu 2020). 
    
    Arguments
    ---------
    xdata : array-like
        The x-coordinates, energy.
    ydata : array-like
        The y-coordinates, intensity.
    fitting_region : tuple
        The (start, end) energy range to fit the Gaussians.
    sat_masks : list of tuples
        List of tuples specifying ranges to mask (e.g., [(start1, end1), (start2, end2)]).
    
    Returns
    -------
    popt : array 
        Optimal values for the parameters.
    pcov : 2-D array
        The estimated covariance of popt.
    
    """

    if fitting_region is None:
        xcropped = xdata
        ycropped = ydata
    else:
        # First crop to specified fitting region
        xcropped = xdata[(xdata >= fitting_region[0]) & (xdata <= fitting_region[1])]
        ycropped = ydata[(xdata >= fitting_region[0]) & (xdata <= fitting_region[1])]
    # Then apply masks to lower energy side of emission lines (satellites)
    sat_mask_bool = np.zeros(xdata.size, dtype=bool)
    for start, end in sat_masks:
        sat_mask_bool |= (xdata >= start) & (xdata <= end)
    xmasked = xcropped[~sat_mask_bool]
    ymasked = ycropped[~sat_mask_bool]

    # Guesses and fitting bounds 
    guess_heb = [ydata.max(), 3683, 1.0]
    guess_lyb = [ydata.max(), 3935, 1.0]
    guess_k   = [0.]
    # Initial guesses [a1, c1, s1, a2, c2, s2, k]
    total_guess = guess_heb + guess_lyb + guess_k

    # defining fitting bounds
    lower_bounds =    [0,-np.inf, 1e-6,]
    upper_bounds =    [ydata.max(), np.inf, 100,]
    lower_bound_k = [-1e-6]
    upper_bound_k = [1e-6]
    # lower bounds for a1, c1, s1, a2, c2, s2, k
    total_lower_bounds = lower_bounds + lower_bounds + lower_bound_k 
    total_upper_bounds = upper_bounds + upper_bounds + upper_bound_k

    # Two Gaussians in one curve_fit 
    popt, pcov = curve_fit(
        fitfunc_two_gaussians,
        xmasked,
        ymasked,
        p0=total_guess,
        bounds=(total_lower_bounds, total_upper_bounds))
    
    #return popt, pcov    
    heb      = popt[0:3]
    heb_pcov = pcov[0:3, 0:3]
    lyb      = popt[3:6]
    lyb_pcov = pcov[3:6, 3:6]

    heb_lw, heb_lw_err, heb_area, heb_area_err = get_fwhm_and_area(*heb, pcov=heb_pcov, R=150)
    lyb_lw, lyb_lw_err, lyb_area, lyb_area_err = get_fwhm_and_area(*lyb, pcov=lyb_pcov, R=150)

    avg_lw = (heb_lw + lyb_lw) / 2
    avg_lw_err = np.sqrt(heb_lw_err**2 + lyb_lw_err**2)

    ratio = lyb_area / heb_area
    ratio_err = ratio * np.sqrt( (lyb_area_err/lyb_area)**2 + (heb_area_err/heb_area)**2 ) # propagate fractional errors then obtain absolute error in line ratio

    # Obtain temperature (assuming n_e = 1e24 cm-3) and density (assuming T_e = 1000 eV) from lookup tables (Keane 1993, Gu 2020)
    lookup_T = np.loadtxt(Path("data/lineratio/lineratio_1e24_keane1993.txt"), delimiter=",").T
    lookup_n = np.loadtxt(Path("data/lineratio/linewidth_Lybeta_1000eV_gu2020.txt"), delimiter=",").T

    temp = np.interp(ratio, lookup_T[1], lookup_T[0])
    dens = np.interp(lyb_lw, lookup_n[1], lookup_n[0])

    # Plotting 
    if plot:
        fig, ax = plt.subplots(nrows=1, figsize=(8,6), sharex=True)
        ax.plot(xdata, ydata, label="Spectrum", color="black")
        ax.axvspan(np.nan, np.nan, color="black", alpha=0.1, label="Satellite Masks")
        for start, end in sat_masks:
            ax.axvspan(start, end, color="black", alpha=0.1)
        xfit = np.linspace(fitting_region[0], fitting_region[1], 1000)
        yfit = fitfunc_two_gaussians(xfit, *popt)
        ax.plot(xfit, yfit, label="Gaussian fits", color="red")
        ax.set_xlim(fitting_region[0]-60, fitting_region[1]+200)
        ax.set_ylabel("Intensity (arb.)")
        ax.legend()
        ax.set_xlabel("Energy (eV)")
        text = f"Ratio = {ratio:.1f} ± {ratio_err:.1f}\nHeβ LW = {heb_lw:.1f} ± {np.ceil(heb_lw_err):.0f} eV\nLyβ LW = {lyb_lw:.2f} ± {np.ceil(lyb_lw_err):.0f} eV\nT$_e$ = {temp:.0f} eV, n$_e$ = {dens*1e-24:.1f} $\\times 10^{{24}}$ cm$^{{-3}}$"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11, horizontalalignment='left', verticalalignment='top')
        fig.subplots_adjust(hspace=0.)

    print(f"Ratio = {ratio:.2f} ± {ratio_err:.2f}, Lybeta LW = {lyb_lw:.2f} ± {lyb_lw_err:.2f}\nT = {temp:.0f} eV, n_e = {dens:.2e} cm^-3")

    if round:
        return np.round(temp, decimals=0), np.round(dens*1e-24, decimals=2)*1e24
    else:
        return temp, dens

if __name__ == "__main__":
    from spec import continuum_butterworth, calibrate_x

    filepath_exp = Path("data/exp/98252_xrf4/sis_f3/sis_f3_no_cr.txt")
    filepath_sigma = filepath_exp.parent / (filepath_exp.stem + "_sigma" + filepath_exp.suffix)

    data_exp = pd.read_csv(filepath_exp, sep="\\s+", header=None)
    data_exp[2] = np.loadtxt(filepath_sigma).T[1]
    data_crop = (3430,5000)
    data_exp[0]    = calibrate_x(data_exp[1], ref_eV=[3683,3934,4150], ref_idx=[143.5, 252.41, 332.65]) # 98252 t4 f3
    data_exp_cropped = data_exp[(data_exp[0]>data_crop[0]) & (data_exp[0]<data_crop[1])].reset_index(drop=True)

    xdata, ydata, ysigma = data_exp_cropped[0], data_exp_cropped[1], data_exp_cropped[2]
    ydata_s = ydata - continuum_butterworth(xdata, ydata, multiplier=1., masks=[(3560,3780),(3790,4055),(4065,4300)])

    get_te_ne(xdata, ydata_s, fitting_region=(3450, 4100), sat_masks=[(3550,3660), (3790,3905)], plot=True)
    plt.show()
