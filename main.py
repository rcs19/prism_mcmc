import numpy as np
import pandas as pd
import corner
import emcee

from pathlib import Path
from matplotlib import pyplot as plt
from lmfit import minimize, Parameters, Minimizer, report_fit

from src.spec import continuum_butterworth, gaussian_broadening, calibrate_x
from src.lineratio import get_te_ne

if __name__ == "__main__":

    # 1. Load experimental data #

    filepath_exp = Path("data/exp/98252_xrf4/sis_f3/sis_f3_no_cr.txt")
    filepath_sigma = filepath_exp.parent / (filepath_exp.stem + "_sigma" + filepath_exp.suffix)

    data_exp = pd.read_csv(filepath_exp, sep="\\s+", header=None)
    data_exp[2] = np.loadtxt(filepath_sigma).T[1]
    data_crop = (3430,5000)
    data_exp[0]    = calibrate_x(data_exp[1], ref_eV=[3683,3934,4150], ref_idx=[143.5, 252.41, 332.65]) # 98252 t4 f3
    data_exp_cropped = data_exp[(data_exp[0]>data_crop[0]) & (data_exp[0]<data_crop[1])].reset_index(drop=True)

    # 2a. Lineratio Analysis for initial T_c and n_c (core electron temperature and density)
    ysubtr = data_exp_cropped[1] - continuum_butterworth(data_exp_cropped[0], data_exp_cropped[1])
    tc, nc = get_te_ne(data_exp_cropped[0], ysubtr, plot=False)

    # 2b. Use rad-hydro for initial n_s and rhoR 
    rho = 25    # g/cm3
    rhoR = 0.09  # g/cm2

    