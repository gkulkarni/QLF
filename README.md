QLF
=====

Homogenized AGN catalogue and luminosity function analysis code from Kulkarni et al. 2018.

-----

This repository contains three things:
1. A homogeneous catalogue of 83,488 AGN between redshift 0 and 7.5, with rest-frame UV magnitudes, redshifts, and selection probabilities.
2. Code to derive luminosity functions from this data 
3. Code to model hydrogen and helium reionization 

The commit tagged `v3.0` reproduces the results from Kulkarni et al. 2018.

The `Data_new` subdirectory contains the AGN catalogue in ASCII text files.  Comments in the files explain their structure.  See Section 2 of the paper for details on the original references for these data and how we homogenise them.  Use `data.py` to visualise the full catalogue in the style of Figure 1 of the paper. Luminosity functions in bins of redshift and magnitude are implemented in `drawlf.py`.  These are shown as points in, e.g., Figure 3 of the paper. Detailed definitions are in Section 3.1 of the paper.Double-power-law luminosity function models are derived in the `lf` class, defined in `individual.py`.  See `lfi.py` or `bins.py` for examples on how to use this class. The latter is used by `mosaic.py` to produce Figure 3 of the paper. Details of this modelling is in Section 3.2 of the paper. Global models of luminosity function evolution are implemented in `composite.py`.  See `lfg.py` or `lfg_multiple.py` for examples of how to use these. Three such models are discussed in Section 3.3 of the paper. Hydrogen-ionizing emissivity is modelled in `gammapi.py`.  The hydrogen-photoionization rate is modelled in `rtg2.py`.  Code in `qhe.py` models Helium reionization. The methods behind these codes is discussed in Section 4 of the paper. 

-------

### FAQ

#### 1. Where do I start?

Begin by running lfi.py to get the luminosity function in a redshift bin.  

#### 2. How do I get the number density of AGN at a certain redshift? 

Pass luminosity function models (instances of `individual.lf` or `composite.lf`) to one of the functions in `rhoqso.py`.

#### 3. I just want to know the value of one of the double-power-law parameters at a redshift

Pass an instance of `composite.lf` to `paramz.parameter_atz()`.

-------

Girish Kulkarni (kulkarni@ast.cam.ac.uk)
