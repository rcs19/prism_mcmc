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

def write_psi_spherical_core(tc, nc, rc, filepathoutpsi="core_spherical_atbase.psi"):
    """
    Write a new PrismSPECT input deck .psi for a spherical core plasma with the given temperature, density and size.
    
    Parameters:
    -----------
    tc: float
        Electron temperature in eV
    nc: float
        Electron density in cm^-3
    rc: float
        Plasma size (radius) in cm 
    template_psi: str or Path
        Path to the template .psi file to use. Default is "data/templates/core_spherical_atbase.psi".
    out_psi: str or Path
        Path to write the new .psi file to. Default is "core_spherical_atbase.psi" in the current working directory.

    Returns:
    --------
    out_psi: str or Path
        Path to the new .psi file that was written.
    """
    with open(template_psi,"r") as f:
        temppsi = json.load(f)
    newpsi = temppsi
    newpsi["Steady-state plasma params"]["Plasma temperature"] = tc
    newpsi["Steady-state plasma params"]["Plasma density"] = nc
    newpsi["Steady-state plasma params"]["Plasma size"] = rc

    with open(out_psi,'w') as f:
        json.dump(newpsi, f, indent=4)

    return out_psi

def write_psi_planar_shell(ts, ns, rhoR, template_psi="data/templates/shell_planar_atbase_rhoR.psi", out_psi="shell_planar_atbase.psi"):
    """
    Write a new PrismSPECT input deck .psi for a planar shell plasma with the given temperature, mass density and areal density rhoR.
    
    Parameters:
    -----------
    ts: float
        Electron temperature in eV
    ns: float
        Mass density in g cm^-3
    rhoR: float
        Areal density in g cm^-2
    template_psi: str or Path
        Path to the template .psi file to use. Default is "data/templates/shell_planar_atbase_rhoR.psi".
    out_psi: str or Path
        Path to write the new .psi file to. Default is "shell_planar_atbase.psi" in the current working directory.

    Returns:
    --------
    out_psi: str or Path
        Path to the new .psi file that was written.
    """
    with open(template_psi,"r") as f:
        temppsi = json.load(f) # note in line 39 "Size specification ID" = 0 for thickness l and = 1 for areal density rhoL
    newpsi = temppsi
    newpsi["Steady-state plasma params"]["Plasma temperature"] = ts
    newpsi["Steady-state plasma params"]["Plasma density"] = ns
    newpsi["Steady-state plasma params"]["Plasma size"] = rhoR

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
        Delete auxiliary files runname.etd .log .psc lineprof.dat popul.pop transpwr.dat. Leaves behind only runname.psr and results/spect.ppd 
    verbose : bool, default = False
        If True, print detailed information about the simulation process.

    Returns
    -------
    output_dir : Path
        Path to the directory containing the PrismSPECT output files (runname.psr and results/spect.ppd)
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
    
    # Run PrismSPECT simulation with hidden outputs (stdout and stderr)
    if verbose:
        subprocess.run(run_command, shell=True)
    else:
        subprocess.run(run_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    output_dir = (psi_filepath.parent / run_name) if run_name is not None else (psi_filepath.parent)

    if delete_aux: # Deletes /runname.etd, /runname.log, /results/lineprof.dat, /results/popul.pop, /results/transpwr.dat    
        delete_command = f'find {output_dir}'+ r' -type f \( -name "*.etd" -o -name "*.log" -o -name "*.dat" -o -name "*.pop" \) -delete'
        subprocess.run(delete_command, shell=True)
        subprocess.run(f'rm {psi_filepath}', shell=True)

    return output_dir

def reduced_model(tc, nc, rc, ts, ns, rhoR, directory=None, runname=None, delete_prism=False, verbose=False):
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
    ts: float
        Electron temperature in eV
    ns: float
        Mass density in g cm^-3
    rhoR: float
        Areal density in g cm^-2
    directory: str or Path, default = None
        Directory to save output files. If None, defaults to "data/".
    runname: str, default = None
        Name for output files and output folder. If None, defaults to a timestamp "yyyymmdd_HHMMSS".
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
        directory = "data/"
    if runname is None:
        runname = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    directory = Path(directory)
    rundirectory = directory / runname
    rundirectory.mkdir(parents=True, exist_ok=True)

    # Generate temporary PrismSPECT input deck .psi for core and shell.
    psi_core_path = write_psi_spherical_core(tc, nc, rc, out_psi=rundirectory / "temp_core.psi")
    psi_shell_path = write_psi_planar_shell(ts, ns, rhoR, out_psi=rundirectory / "temp_shell.psi")

    # This will now run PrismSPECT twice, first for the core simulation then the shell simulation. The output files are:
    # outputfolder/runname/runname.psc - copy of input deck
    # outputfolder/runname/runname.psr - main results file
    # outputfolder/runname/results/spect.ppd - output spectra 
    if verbose:
        print(f"Running PrismSPECT with params:\ntc={tc}, nc={nc}, rc={rc}, ts={ts}, ns={ns}, rhoR={rhoR}")
    core_path = run_PrismSPECT(psi_core_path, run_name="temp_core", overwrite=True, delete_aux=True, verbose=verbose)
    shell_path = run_PrismSPECT(psi_shell_path, run_name="temp_shell", overwrite=True, delete_aux=True, verbose=verbose)

    # Load spectra and transmission from output files spect.ppd
    core_ppd = np.loadtxt(core_path / "results/spect.ppd", comments="#").T
    shell_ppd = np.loadtxt(shell_path / "results/spect.ppd", comments="#").T
    nu_core, I_core = core_ppd[0], core_ppd[1]
    nu_shell, I_shell, op_shell = shell_ppd[0], shell_ppd[1], shell_ppd[2]
    tr_shell = np.exp ( -op_shell * rhoR )

    # Interpolate shell onto core nu grid
    I_shell_interp = np.interp(nu_core, nu_shell, I_shell)
    tr_shell_interp = np.interp(nu_core, nu_shell, tr_shell)

    # Calculate emergent intensity distribution
    eid_y = I_core * tr_shell_interp + I_shell_interp
	
    # Save emergent intensity distribution to file
    eid = np.array([nu_core, eid_y]).T
    np.savetxt(directory / f"{runname}.txt", eid)

    if delete_prism:
        subprocess.run(f'rm -r {directory}/{runname}', shell=True)

    if False:
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(nu_core, I_core, label="Core spec")
        ax[0].plot(nu_shell, I_shell, label="Shell spec")
        ax_trans = ax[0].twinx()
        ax_trans.plot(nu_shell, tr_shell, color="black", ls="--")
        ax[1].plot(nu_core, eid_y, label="EID")
        ax[1].plot(nu_shell, I_shell, label="Shell spec")
        ax[0].legend()
        ax[1].legend()
        plt.show()

    return eid

if __name__ == "__main__":
    eid = reduced_model(tc=1000, nc=1e24, rc=40e-4, ts=400, ns=25, rhoR=0.09, directory = "data/20260311/", runname = "sample1", delete_prism=True)
