"""
Functions for 
- Generating PrismSPECT workspace files .psc (input decks)
- Running PrismSPECT, deleting unnecessary files (to save space) and grabbing spectrum output
"""

import json
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import time

def write_psi(temp, dens, size, carbonmix=None, template_psi=None, out_psi="test.psi"):
    """
    Write a new PrismSPECT input deck .psi for a spherical core plasma with the given temperature, density and size.
    The units of density and size depend on the given template.psi.
    For 
    
    Parameters:
    -----------
    temp: float
        Electron temperature in eV
    dens: float
        Denisty, units defined in template.psi. For core.psi this is electron density in cm^-3. For shell.psi this is mass density g cm^-3
    size: float
        Plasma size, units defined in template.psi. For core.psi this is spherical radius in cm. For shell.psi this is areal density in g cm^-2
    template_psi: str or Path
        Path to the template .psi file to use. Default is "data/templates/core_spherical_atbase.psi".
    out_psi: str or Path
        Path to write the new .psi file to. Default is "test.psi" in the current working directory.

    Returns:
    --------
    out_psi: str or Path
        Path to the new .psi file that was written.
    """
    try:
        template_psi = Path(template_psi)
    except:
        raise ValueError("template_psi must be a valid path to a .psi file.")

    with open(template_psi,"r") as f:
        temppsi = json.load(f)
    newpsi = temppsi
    newpsi["Steady-state plasma params"]["Plasma temperature"] = temp
    newpsi["Steady-state plasma params"]["Plasma density"] = dens
    newpsi["Steady-state plasma params"]["Plasma size"] = size

    if carbonmix is not None:
        newpsi["Material params"]["Atomic element[3]"]["Number fraction"] = carbonmix

    with open(out_psi,'w') as f:
        json.dump(newpsi, f, indent=4)

    return out_psi

def run_PrismSPECT(psi_filepath, run_name=None, overwrite=True, delete_aux=True, verbose=False):
    """
    Run PrismSPECT with the given input deck .psi file, and optionally delete auxiliary files to save space.
    The output folder containing results files is placed in the same folder as the input deck (-d {output_dir} does not work).

    Parameters
    ----------
    psi_filepath: str or Path
        Path to the .psi input deck file for PrismSPECT
    run_name : str, default = None
        Name for output files and output folder. If run_name is not given, the name of the output folder and output files defaults to the name of the input file (without extension)
    overwrite : bool, default = True
        Overwrite existing output of the same name
    delete_aux : bool, default = True
        Delete auxiliary files run_name.etd .log .psc lineprof.dat popul.pop transpwr.dat. Leaves behind only run_name.psr and results/spect.ppd 
    verbose : bool, default = False
        If True, print detailed information about the simulation process.

    Returns
    -------
    output_dir : Path
        Path to the directory containing the PrismSPECT output files (run_name.psr and results/spect.ppd)
    """
    psi_filepath = Path(psi_filepath)

    run_command = f"PrismSPECT -b -i {psi_filepath}"
    if run_name is not None:
        run_command = run_command + f" -o {run_name}"
        runfolder = Path(psi_filepath.parent / run_name).mkdir(exist_ok=True)
    # if output_dir is not None: 
    #     run_command = run_command + f" -d {output_dir}" # !! doesn't work - "Run directory could not be created. Check permissions or see if directory is in use."
    if overwrite:
        run_command = run_command + " -x"
    
    ## Run PrismSPECT simulation with hidden outputs (stdout and stderr)
    # if verbose:
    #     subprocess.run(run_command, shell=True)
    # else:
    #     subprocess.run(run_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run(run_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) # force run silent

    output_dir = (psi_filepath.parent / run_name) if run_name is not None else (psi_filepath.parent)

    if delete_aux: # Deletes /run_name.etd, /run_name.log, /results/lineprof.dat, /results/popul.pop, /results/transpwr.dat    
        delete_command = f'find {output_dir}'+ r' -type f \( -name "*.etd" -o -name "*.log" -o -name "*.dat" -o -name "*.pop" \) -delete'
        subprocess.run(delete_command, shell=True)
        subprocess.run(f'rm {psi_filepath}', shell=True)

    return output_dir

def reduced_model(tc, nc, rc, ts, ns, rhoR, carbonmix=None, corepsi="data/inputs/templates/core_spherical_atbase.psi", shellpsi="data/inputs/templates/shell_planar_atbase_rhoR.psi", directory=None, run_name=None, reuse_run=None, overwrite=False, delete_prism=False, verbose=False):
    """
    Runs the reduced spherical core + shell model. 
    Wrapper function for running core simulation and shell simulations to obtain
        - Core spectrum 
        - Shell spectrum (shell self-emission)
        - Shell transmission
    
    Calculates and returns the emergent intensity distribution f(nu) = I_core(nu)*T_shell(nu) + I_shell(nu).
    The emergent intensity distribution can be broadened using a gaussian broadening function with R = E/dE, if necessary.  

    Parameters
    ----------
    tc: float
        Electron temperature in eV
    nc: float
        Electron density in cm^-3
    rc: float
        Plasma size (radius) in cm
    carbonmix: float, optional
        Carbon number fraction
    ts: float
        Electron temperature in eV
    ns: float
        Mass density in g cm^-3
    rhoR: float
        Areal density in g cm^-2
    directory: str or Path, default = None
        Directory to save output files. If None, defaults to "data/".
    run_name: str, default = None
        Name for output files and output folder. If None, defaults to a timestamp "yyyymmdd_HHMMSS".
    reuse_run: str or list, default = None
        Reuse existing output files from specified name of previous run. Will check if "directory/reuse/sample*.txt" exists first before checking current run directory (if overwrite=True). Otherwise run as normal, running new simulations.
    overwrite: bool, default = False
        If False, checks if "directory / run_name_eid.txt" exists and reuses it. Otherwise, re-run PrismSPECT simulations. If True, PrismSPECT will run anyway. !Caution! The reused output may not correspond to the given input, so only use if certain. 
    delete_prism: bool, default = False
        If True, deletes the output folder containing PrismSPECT results after extracting the emergent intensity distribution. 
    verbose: bool, default = False
        Prints statements from this script and also PrismSPECT outputs.

    Returns
    -------
    eid: 2D array
        Emergent intensity distribution of attenuated core emissions + shell self emission. A 2D array with columns [nu, f(nu)]
    """

    if directory is None:
        directory = "data/mcmc_unnamed_run/"
    if run_name is None:
        run_name = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    directory = Path(directory)

    if reuse_run is not None:
        reuse_directories = []
        if isinstance(reuse_run, str):
            reuse_directories.append(reuse_run)
        elif isinstance(reuse_run, list):
            reuse_directories = [Path(r) for r in reuse_run]
        
        for rundir in reuse_directories:
            rundir = Path(rundir)
            if (rundir / f"{run_name}_eid.txt").exists():
                if verbose:
                    print(f"Reusing {rundir} / {run_name}_eid.txt...")
                nu, eid, bf = np.loadtxt(rundir / f"{run_name}_eid.txt", unpack=True)
                return nu, eid

    if not overwrite:
        # Check if directory / run_name_eid.txt exists
        if (directory / f"{run_name}_eid.txt").exists():
            if verbose:
                print(f"{directory / run_name}_eid.txt exists - reusing...")
            nu, eid, bf = np.loadtxt(directory / f"{run_name}_eid.txt", unpack=True)
            return nu, eid
    
    attempts = 0
    while attempts < 3:
        try:
            rundirectory = directory / run_name
            rundirectory.mkdir(parents=True, exist_ok=True)

            # Generate temporary PrismSPECT input deck .psi for core and shell.
            psi_core_path = write_psi(tc, nc, rc, carbonmix, template_psi=corepsi, out_psi=rundirectory / "temp_core.psi")
            psi_shell_path = write_psi(ts, ns, rhoR, template_psi=shellpsi, out_psi=rundirectory / "temp_shell.psi")

            # This will now run PrismSPECT twice, first for the core simulation then the shell simulation. The output files (not deleted by delete_aux):
            # directory/run_name/run_name.psc - copy of input deck
            # directory/run_name/run_name.psr - main results file
            # directory/run_name/results/spect.ppd - output spectra 
            if verbose:
                carbonmix_str = f"{carbonmix:.2f}" if carbonmix is not None else "None"
                print(f"Running PrismSPECT with params:\ntc={tc:.0f}, nc={nc:.4g}, rc={rc}, C_mix={carbonmix_str}, ts={ts:.1f}, ns={ns:.1f}, rhoR={rhoR:.4f}")
            core_path = run_PrismSPECT(psi_core_path, run_name="temp_core", overwrite=True, delete_aux=True, verbose=verbose)
            shell_path = run_PrismSPECT(psi_shell_path, run_name="temp_shell", overwrite=True, delete_aux=True, verbose=verbose)

            # Load spectra and transmission from output files spect.ppd
            nu_core, I_core, bf_core, ff_core = np.loadtxt(core_path / "results/spect.ppd", comments="#", usecols=(0,1,12,14), unpack=True)
            nu_shell, I_shell, op_shell = np.loadtxt(shell_path / "results/spect.ppd", comments="#", usecols=(0,1,2), unpack=True) 
            break
        except FileNotFoundError as e:
            attempts += 1
            print(f"!Error: {e} attempt {attempts}")
            time.sleep(1) # Wait a bit before retrying
            if attempts >= 3:
                raise e

    tr_shell = np.exp ( -op_shell * rhoR )

    # Interpolate shell onto core nu grid
    I_shell_interp = np.interp(nu_core, nu_shell, I_shell)
    tr_shell_interp = np.interp(nu_core, nu_shell, tr_shell)

    # Calculate emergent intensity distribution
    bf_core = bf_core * tr_shell_interp
    ff_core = ff_core * tr_shell_interp
    eid_y = I_core * tr_shell_interp + I_shell_interp
	
    # Save emergent intensity distribution to file
    np.savetxt(directory / f"{run_name}_eid.txt", X=np.column_stack([nu_core, eid_y, bf_core, ff_core]), fmt="%.7e")

    if delete_prism:
        subprocess.run(f'rm -r {directory}/{run_name}', shell=True)

    return nu_core, eid_y

if __name__ == "__main__":
    core_psi_path = write_psi(1000, 1e22, 0.04, carbonmix=0.2753, template_psi="data/inputs/templates/core_spherical_atbase_leastdetailed_DArC.psi", out_psi="test_core.psi")