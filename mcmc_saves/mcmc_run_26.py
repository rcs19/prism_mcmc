filepath = "data/exp/98252_xrf3_Mar2026/sis_f1/sis_f1_no_cr.txt"
ref_eV  = [3420,3683,3934,4152]
ref_idx = [0., 107.7, 219.1, 302]
weights = {"centers":[3665, 3930], "sigmas":[60,60], "amplitude":5, "baseline":1, "n":4}

# 1b. Define parameters, initial guess, bounds and MCMC settings
params_initial = {'tc_kev': 1.0, 
                  'lognc': 23.78, 
                  'carbonmix': 0.15,
                  'ts_kev': 0.28, 
                  'rhoR': 0.13}

params_bounds  = {'tc_kev': (0.7, 1.4), 
                  'lognc': (23.0, 24.5), 
                  'carbonmix': (0.0, 0.4),
                  'ts_kev': (0.1, 0.6), 
                  'rhoR': (0.04, 0.17)}

fitting_mask   = [(3560,3740), (3840,4020), (4070,4500)]
nwalkers       = 12
nsteps         = 120
corepsi        = "data/inputs/templates/core_spherical_atbase_leastdetailed_DArC.psi"
shellpsi       = "data/inputs/templates/shell_planar_atbase_rhoR.psi"
directory      = "data/mcmc_run_26/"
savefile       = "mcmc_saves/mcmc_run_26.h5"
reuse_run      = None 
verbose        = True    