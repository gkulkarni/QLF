QLF data compilation (QSO lists and selection functions)
========================================================
- All files in same format and units, same cosmology (Omega_M=0.3, Omega_Lambda=0.7, H0=70km/s/Mpc)
  QSO samples list counter, redshift, absolute magnitude at rest wavelength 1450A, selection probability, survey area in deg2 and sample ID
  QSO selection functions list counter, redshift, absolute magnitude at rest wavelength 1450A and selection probability
- All selection functions corrected for other sources of incompleteness mentioned in the papers but not incorporated into selection functions by default

Sample IDs from previous analysis on binned data, not all of these are used in the unbinned analysis
- ID=1	   BOSS DR9 2.2<z<3.5 color-selected sample (Ross et al. 2013, ApJ, 773, 14)
- ID=6	   NDWFS+DLS 3.7<z<5.2 (Glikman et al. 2011, ApJ, 728, L26),
           separate selection functions for the two fields, debugged QSO list
- ID=7	   CANDELS GOODS-S 4.0<z<6.5 (Giallongo et al. 2015, A&A, 578, A83),
           estimated selection function, many photometric redshifts
- ID=8	   SDSS DR7+Stripe82 4.7<z<5.1 (McGreer et al. 2013, ApJ, 768, 105),
           separate samples and selection functions, DR7 selection function supersedes Richards et al. 2006,
           extended sample (5.1<z<5.5) is experimental (low selection probability, not included in McGreer et al.)
- ID=9	   SDSS Deep z~6 survey (Jiang et al. 2008, AJ, 135, 1057; Jiang et al. 2009, AJ, 138, 305),
           different survey areas, depths and selection functions, selection functions provided by Chris Willott
- ID=10	   CFHQS Very Wide+Deep z~6 surveys (Willott et al. 2010, AJ, 139, 906),
           separate samples and selection functions
- ID=11	   Subaru z~6 survey in UKIDSS-DXS (Kashikawa et al. 2015, 798, 28),
	   both discovered objects included (one has quite narrow Lyman alpha emission, LBG???)
- ID=13	   SDSS DR7 (uniform sample) at z<2.2 and z>3.7 to avoid color selection bias (Richards et al. 2006, Schneider et al. 2010),
	   separate samples and selection functions,
	   z<2.2 sample corrected for host galaxy light (Croom et al. 2009, MNRAS, 392, 19),
           use with caution at z<0.4 due to uncertainty in host galaxy light correction,
	   K correction from i band uncertain at z>4.7 (Lya emission, Lya forest) but selection map given in i band
           -> use only at z<4.7 (also to avoid duplicates with McGreer et al. 2013 and Yang et al. 2016)
- ID=15	   2SLAQ 0.4<z<2.6 gmag_dered<21.85 (Croom et al. 2009, MNRAS, 392, 19; Croom et al. 2009, MNRAS, 399, 1755),
	   separate samples and selection functions for NGP and SGP fields,
	   all magnitudes corrected for host galaxy light,
	   redshift range 2.2<z<2.6 may be affected by color bias (outdated selection function), likely superseded by BOSS
	   -> restrict this sample to z<2.2
- ID=16	   SDSS Main z~6 survey (Fan et al. 2006, AJ, 131, 1203),
           selection function provided by Chris Willott 
- ID=17	   SDSS+WISE z~5 survey (Yang et al. 2016, ApJ, 829, 33),
           4.7<=z<5.4, significant overlap with SDSS DR7 QSO catalog (i.e. Sample 8 and 13)
           -> limit 4.7<z<5.4 SDSS DR7 QSOs to M1450>-26.73 (our cosmology) when combining samples to avoid double-counting
- ID=18	   SDSS z~6 survey compilation (Jiang et al. 2016, ApJ, 833, 222),
           different survey areas (Main, Overlap, Stripe82), depths and selection functions, selection functions provided by Linhua Jiang
           -> supersedes previous SDSS sample (Sample 9 and 16)