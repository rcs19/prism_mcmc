"""
Functions for 
- Generating PrismSPECT workspace files .psc (input decks)
- Running PrismSPECT, deleting unnecessary files (to save space) and grabbing spectrum output
"""

import json
import subprocess
from pathlib import Path

def write_psi_spherical_core(tc, nc, r, filepathout="core_spherical_atbase.psi"):
    """
    Write a new PrismSPECT input deck .psi for a spherical core plasma with the given temperature, density and size.
    
    Parameters:
    -----------
    tc: float
        Plasma temperature in eV
    nc: float
        Plasma density in cm^-3
    r: float
        Plasma size (radius) in cm
    """
    with open('data/inputs/templates/core_spherical_atbase.psi','r') as f:
        temppsi = json.load(f)
    newpsi = temppsi
    newpsi["Steady-state plasma params"]["Plasma temperature"] = tc
    newpsi["Steady-state plasma params"]["Plasma density"] = nc
    newpsi["Steady-state plasma params"]["Plasma size"] = r

    with open(filepathout,'w') as f:
        json.dump(newpsi, f, indent=4)

    return filepathout

def write_psi_planar_shell(ts, ns, rhoR, filepathout="shell_planar_atbase.psi"):
    """
    Write a new PrismSPECT input deck .psi for a planar shell plasma with the given temperature, mass density and areal density rhoR.
    
    Parameters:
    -----------
    ts: float
        Plasma temperature in eV
    ns: float
        Plasma density in cm^-3
    rhoR: float
        Plasma size (radius) in cm
    """
    with open('data/inputs/templates/shell_planar_atbase_rhoR.psi','r') as f:
        temppsi = json.load(f) # note in line 39 "Size specification ID" = 0 for thickness l and = 1 for areal density rhoL
    newpsi = temppsi
    newpsi["Steady-state plasma params"]["Plasma temperature"] = ts
    newpsi["Steady-state plasma params"]["Plasma density"] = ns
    newpsi["Steady-state plasma params"]["Plasma size"] = rhoR

    with open(filepathout,'w') as f:
        json.dump(newpsi, f, indent=4)

    return filepathout

def run_PrismSPECT(psi_filepath, run_name=None, overwrite=True, delete_aux=True):
    """
    Run PrismSPECT with the given input deck .psi file, and optionally delete auxiliary files to save space.
    The output folder containing results files is placed in the same folder as the input deck (-d {output_dir} does not work).

    Parameters
    ----------
    psi_filepath: str or Path
        Path to the .psi input deck file for PrismSPECT
    run_name : str, default = None
        Name for output files and output folder. If run_name is not given, the name of the output folder and output files defaults to the   
    overwrite : bool, default = True
        Overwrite existing output of the same name
    delete_aux : bool, default = True
        Delete auxiliary files runname.etd .log .psc lineprof.dat popul.pop transpwr.dat. Leaves behind only runname.psr and results/spect.ppd 
    """
    psi_filepath = Path(psi_filepath)

    run_command = f"PrismSPECT -b -i {psi_filepath}"
    if run_name is not None:
        run_command = run_command + f" -o {run_name}"
    # if output_dir is not None: 
    #     run_command = run_command + f" -d {output_dir}" # !! doesn't work - "Run directory could not be created. Check permissions or see if directory is in use."
    if overwrite:
        run_command = run_command + " -x"

    # Run PrismSPECT simulation
    subprocess.run(run_command, shell=True)

    if delete_aux:
        # Deletes /runname.etd 35 MB, /runname.log, /runname.psc /results/lineprof.dat 24 MB, /results/popul.pop 1 MB, /results/transpwr.dat 13 MB
        output_dir = (psi_filepath.parent / run_name) if run_name is not None else (psi_filepath.parent)
        delete_command = f'find {output_dir}'+ r' -type f \( -name "*.etd" -o -name "*.log" -o -name "*.psc" -o -name "*.dat" -o -name "*.pop" \) -delete'
        subprocess.run(delete_command, shell=True)

if __name__ == "__main__":
    path_shell_psi = write_psi_planar_shell(300,3e24,0.095, filepathout="archive/test_planar_shell.psi")
    run_PrismSPECT(path_shell_psi, run_name="test_run", overwrite=True, delete_aux=True)