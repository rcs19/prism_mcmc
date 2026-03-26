filepath = "data/exp/98252_xrf4_Mar2026/sis_f3/sis_f3_no_cr.txt"
ref_eV  = [3420,3683,3934,4150]
ref_idx = [0, 144, 253, 332]
weights = None

# 1b. Define parameters, initial guess, bounds and MCMC settings
params_initial = {'tc_kev': 1.08, 
                  'lognc': 24.29, 
                  'carbonmix': 0.13, 
                  'ts_kev': 0.3, 
                  'rhoR': 0.1}

params_bounds  = {'tc_kev': (0.7, 1.4), 
                  'lognc': (23.0, 25), 
                  'carbonmix': (0.01, 0.4), 
                  'ts_kev': (0.1, 0.6), 
                  'rhoR': (0.04, 0.17)}

fitting_mask   = [(3580,3750), (3830,4050),]
nwalkers       = 10
nsteps         = 100
corepsi        = "data/inputs/templates/core_spherical_atbase_leastdetailed_DArC.psi"
shellpsi       = "data/inputs/templates/shell_planar_atbase_rhoR.psi"
directory      = "data/mcmc_run_20/"
savefile       = "mcmc_run_20.h5"
reuse_run      = None 
verbose        = True    