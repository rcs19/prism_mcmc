filepath = "data/exp/98252_xrf3_Mar2026/sis_f1/sis_f1_no_cr.txt"
ref_eV  = [3420,3683,3934,4150]
ref_idx = [0., 107.7, 219.1, 302.7]
weights = {"centers":[3665, 3930], "sigmas":[60,80], "amplitude":10, "baseline":0.5, "n":4}

# 1b. Define parameters, initial guess, bounds and MCMC settings
params_initial = {'tc_kev': 1.08, 
                  'lognc': 24.29, 
                  'ts_kev': 0.3, 
                  'rhoR': 0.1}

params_bounds  = {'tc_kev': (0.7, 1.4), 
                  'lognc': (23.0, 25), 
                  'ts_kev': (0.1, 0.6), 
                  'rhoR': (0.04, 0.17)}

fitting_mask   = [(3450, 4600)]
nwalkers       = 10
nsteps         = 150
corepsi        = "data/inputs/templates/core_spherical_atbase_leastdetailed.psi"
shellpsi       = "data/inputs/templates/shell_planar_atbase_rhoR.psi"
directory      = "data/mcmc_run_22/"
savefile       = "mcmc_run_22.h5"
reuse_run      = None 
verbose        = True    