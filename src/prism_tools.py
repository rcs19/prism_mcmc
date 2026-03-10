"""
Functions for 
- Generating PrismSPECT workspace files .psc (input decks)
- Running PrismSPECT, deleting unnecessary files (to save space) and grabbing spectrum output

Quick notes:
- Delete:
    - /runname.etd 35 MB
    - /results/lineprof.dat 24 MB
    - /results/popul.pop 1 MB
    - /results/transpwr.dat 13 MB

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

def run_PrismSPECT(psi_filepath, run_name, output_dir, delete_aux=True):
    subprocess.run(f'PrismSPECT -b -i {psi_filepath} -x', shell=True)
    
if __name__ == "__main__":
    write_psi_planar_shell(300,3e24,0.095)