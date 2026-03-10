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