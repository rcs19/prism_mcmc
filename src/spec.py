import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

def continuum_butterworth(xdata, ydata, cutoff=5, multiplier=1.0, masks=[(3560,3780),(3790,4055),(4065,4300)]):
    """
    Find a continuum by masking emission lines and filtering/smoothing the remaining spectrum.
    1. Mask specified ranges/emission lines. The data is replaced with a straight line (line joining median intensity around mask start and end).
    2. A butterworth filter is applied to smooth the data.

    Parameters
    ----------
    xdata : array-like
        The x-coordinates, energy.
    ydata : array-like
        The y-coordinates, intensity.
    cutoff : int
        The cutoff frequency for the Butterworth filter.
    multiplier : float
        Multiplier to adjust amplitude of computed continuum.
    masks : list of tuples, optional
        List of tuples specifying ranges to mask (e.g., [(start1, end1), (start2, end2)]).
    
    Returns
    -------
    y_continuum : array-like
        The computed continuum.

    """
    # Interpolate x-grid to ensure even spacing
    xinterp = np.linspace(xdata.min(), xdata.max(), len(xdata)*5) # oversample
    yinterp = np.interp(xinterp, xdata, ydata)

    # Replace masked regions with a straight line. Use the median of the data around the mask start and end.
    median_width = 10  # Number of points to consider for median calculation around the mask edges
    for start, end in masks:
        mask_indices = (xinterp >= start) & (xinterp <= end)
        if np.any(mask_indices):
            # Get the indices before and after the mask
            before_mask = np.where(xinterp < start)[0] # Indices before the mask. before_mask[-1] is the index of the starting edge of the mask.
            after_mask = np.where(xinterp > end)[0]    # Indices after the mask. after_mask[0] is the index of the ending edge of the mask.
            if before_mask.size > 0 and after_mask.size > 0:
                # Use the median of the data around the mask start and end
                y_start = np.median(yinterp[before_mask[-1] - median_width:before_mask[-1] + median_width]) if before_mask.size > median_width else yinterp[before_mask[-1]]
                y_end = np.median(yinterp[after_mask[0] - median_width:after_mask[0] + median_width]) if after_mask.size > median_width else yinterp[after_mask[0]]
                yinterp[mask_indices] = np.linspace(y_start, y_end, np.sum(mask_indices)) # np.sum(mask_indices) gives the length of the masked region (i.e. np.sum([False, False, True, True, False]) == 2)

    # Butterworth filter (lowpass smoothing filter)
    N = len(yinterp)
    order = 2
    cutoff = 2.0 * cutoff / N                   # Cutoff frequency 
    b, a = butter(order, cutoff, btype='low',)  # scipy.signal.butter returns the filter coefficients b and a
    yinterp = filtfilt(b, a, yinterp)             # Apply the filter to the data    
    yinterp = yinterp * multiplier                # Adjust the amplitude of the computed continuum

    # Interpolate back to original x-grid
    return np.interp(xdata, xinterp, yinterp)

def gaussian_broadening(xdata, ydata, R):
    """
    Apply Gaussian broadening with constant spectral resolving power 
    R = E / DeltaE to data defined on a non-uniform energy grid xdata.
    Returns the broadened ydata on the original xdata grid.
    !!Note I am using numpy.interp instead of scipy.interpolate.interp1d because it is faster. Scipy was originally used to broaden multiple spectra, whereas numpy can only handle one.

    Parameters
    ----------
    xdata : array-like
        Original energy grid (non-uniform)
    ydata : array-like
        Original intensity data
    R : float
        Spectral resolving power (R = E / DeltaE)
    
    Returns
    -------
    y_broadened : array-like
        Broadened intensity data on the original xdata grid
    """
    # Step 1: Convert to log-energy space
    logE = np.log(xdata)

    # Step 2: Interpolate to a uniform grid in log-space (oversampling here with len(xdata)*4)
    logE_uniform = np.linspace(np.min(logE), np.max(logE), len(xdata)*4)
    # interp = interp1d(logE, ydata, kind='linear', fill_value='extrapolate')
    # y_uniform = interp(logE_uniform)
    y_uniform = np.interp(logE_uniform, logE, ydata)

    # Step 3: Compute Gaussian sigma (constant in log-space)
    # For a Gaussian, FWHM = 2*sqrt(2*ln(2))*sigma
    # Here, FWHM (in logE) = ln(1 + 1/R) ≈ 1/R for large R
    FWHM_logE = np.log(1 + 1/R)
    sigma_logE = FWHM_logE / (2 * np.sqrt(2 * np.log(2)))

    # Step 4: Convert sigma_logE to number of grid points
    dlogE = np.mean(np.diff(logE_uniform))
    sigma_points = sigma_logE / dlogE

    # Step 5: Convolve in log-space
    y_broadened_uniform = gaussian_filter1d(y_uniform, sigma_points)

    # Step 6: Interpolate back to original (non-uniform) energy grid
    # interp_back = interp1d(logE_uniform, y_broadened_uniform, kind='linear', fill_value='extrapolate')
    # y_broadened = interp_back(np.log(xdata))
    y_broadened = np.interp(logE, logE_uniform, y_broadened_uniform)
    return y_broadened

def calibrate_x(ydata = None, ref_idx = None, ref_eV = None ):
    """
    Generate an xdata array for ydata and ydata sigma.
    """
    # If reference points are provided, use this to calibrate the xdata. Otherwise, show an interactive plot of the spectrum and ask user to click on the given reference points to calibrate the xdata. If there are no reference points gien, print a warning and return the original xdata as an array of indices.

    # Use linear interpolation/extrapolation to find xdata values that fit on the reference points
    if ref_idx is not None and ref_eV is not None and ydata is not None:
        f = interp1d(ref_idx, ref_eV, fill_value="extrapolate")
        xdata = f(np.arange(len(ydata)))
        return xdata
        
    elif ydata is not None:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(ydata)), ydata, label="Spectrum")
        ax.set_title("Click on reference points in order:\n[" + ", ".join([f"{eV}" for eV in ref_eV]) + "] eV")
        def onclick(event):
            if event.xdata is not None:
                clicked_idx.append(event.xdata)
                ax.plot(event.xdata, event.ydata, marker="x", color="red")
                # draw idle
                fig.canvas.draw_idle()
                if len(clicked_idx) == len(ref_eV):
                    print("[" + ", ".join([f"{i:.2f}" for i in clicked_idx]) + "]")
                    plt.close()
        clicked_idx = []
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        f = interp1d(clicked_idx, ref_eV, fill_value="extrapolate")
        xdata = f(np.arange(len(ydata)))
        return xdata
    else:
        print("Please provide ydata and reference points eV for calibration. Returning original xdata as indices.")
        return np.arange(len(ydata))

def get_ysigma(ydata, window=5):
    ydata_smooth = np.convolve(ydata, np.ones(window)/window, mode='same')
    var = (ydata - ydata_smooth)**2 # residuals of each point with smoothed data
    # variance of point is mean of var within window
    std2 = np.convolve(var, np.ones(window), mode='same')
    count = np.convolve(np.ones_like(ydata), np.ones(window), mode='same')
    count = np.maximum(count,2.)
    ydata_sigma = np.sqrt(std2/(count-1))

    return ydata_sigma

def adjust_weights(xdata, ysigma, regions, multiplier=0.5):
    """
    Adjust the weights (i.e., sigma) of data points in specified regions by multiplying with a given factor.

    Parameters
    ----------
    ysigma : array-like
        The uncertainties (sigma) associated with each data point.
    regions : list of tuples
        List of tuples specifying ranges to adjust (e.g., [(start1, end1), (start2, end2)]).
    multiplier : float
        The factor by which to multiply the sigma values in the specified regions.

    Returns
    -------
    ysigma_adjusted : array-like
        The adjusted sigma values.
    """
    ysigma_adjusted = np.array(ysigma)  
    for start, end in regions:
        mask = (xdata >= start) & (xdata <= end)
        ysigma_adjusted[mask] *= multiplier
    ysigma_adjusted = ysigma_adjusted / multiplier
    
    return ysigma_adjusted


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    # 1. Load experimental data
    # filepath_exp = Path("data/exp/98252_xrf4/sis_f3/sis_f3_no_cr.txt")
    filepath_exp = Path(f"data/exp/98263_xrf4/f3/f3_no_cr.txt")

    filepath_sigma = filepath_exp.parent / (filepath_exp.stem + "_sigma" + filepath_exp.suffix)
    data_exp = pd.read_csv(filepath_exp, sep="\\s+", header=None)
    data_exp[2] = np.loadtxt(filepath_sigma).T[1]
    data_crop = (3430,5000)
    # data_exp[0]    = calibrate_x(data_exp[1], ref_eV=[3683,3934,4150], ref_idx=[143.5, 252.41, 332.65]) # 98252 t4 f3
    data_exp[0]    = calibrate_x(data_exp[1], ref_eV=[3683,3934,4150], ref_idx=[142.24, 254, 339]) # 98263 t4 f3
    data_exp_cropped = data_exp[(data_exp[0]>data_crop[0]) & (data_exp[0]<data_crop[1])].reset_index(drop=True)

    xdata, ydata, ysigma  = data_exp_cropped[0].values, data_exp_cropped[1].values, data_exp_cropped[2].values

    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, label="Data")
    ax.fill_between(xdata, ydata-ysigma, ydata+ysigma, color="gray", alpha=0.5, label="Data sigma")
    # plt.show()

    ysigma_adjusted = adjust_weights(xdata, ysigma, regions=[(3600,3710), (3830,3960), (4060,4400)], multiplier=0.4)
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, label="Data")
    ax.fill_between(xdata, ydata-ysigma_adjusted, ydata+ysigma_adjusted, color="gray", alpha=0.5, label="Data sigma")
    plt.show()