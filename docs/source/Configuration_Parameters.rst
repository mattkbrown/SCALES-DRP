Configuration Parameters
=========================

A number of reduction parameters can be changed using entries in the configuration file.

If you installed the pipeline with ``pip``, the configuration file will not be easy to find,
since it will be stored with installed ``pip`` packages. We recommend editing a copy of the
config instead. You can create a copy of the config file by invoking the pipeline with
the ``--write_config`` option:

.. code-block:: bash

   start_scales_reduce --write_config


.. Note ::

   If a parameter in the ``scales.cfg`` or ``scales_calib.cfg`` file is not described here,
   then you can assume that it should not be modified.

Configuration Parameters Common to both the module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have default parameters for both calibration module and science-grade module operations.
The configuration parameters corresponding to calibration and science-grade module
can be found in  ``scales_calib.cfg`` and ``scales.cfg``, respectively.
The list of common tunable parameters are listed below.

.. code-block:: bash
	clobber = False #Redo the slope image generation even if the products exist
  output_directory = 'redux/' #output directory name

Dectector level corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

  do_swap = False #apply odd even column swapping on the raw reads.
  nchans=4 #Number of readout channel on both the detectors
  altcol=True #ACN correction using odd and even column separately for each channel
  channelwise=True #ACN correction using a single value per channel
  amp_mean_func=robust.mean #ACN correction function
  do_acn=True # ACN 2nd level correction needed?
  acn_avg_type='pix' #'pix' or 'frame'
  acn_mean_func=np.median # ACN 2nd level correction function
  acn_smooth=True #smoothening needed?
  acn_savgol=False #savgol filter for smoothening
  acn_winsize=31 #smoothening window size
  acn_order=3
  resid_colsub=False #enable residual column subtraction
  fixcol=True #enable 1/f correction
  ref_avg_type='row_wise'#'frame', 'pix', 'row_wise'
  ref_mean_func=np.median
  ref_smooth=True
  ref_savgol=False
  ref_winsize=31
  ref_order=3
  pickup=False #enable pickup noise correction
  sigma_thresh=4.0
  dilate_iter=2
  highpass_size=101
  per_amp=False

  do_linearity = True # Do you wanted to apply linearity correction?
  apply_sat_mask=True #Do you like to apply saturation mask to ramp fitting
  apply_bpm = True #Do you like to apply bad pixel mask to the slope image?

  apply_dark=True #Do you like to apply master dark correction if the file exist?
  apply_det_flat=True #Do you like to apply master detector flat correction if the file exist?
  apply_bias=True Do you like to apply master bias correction if the file exist?

Calibration Files associated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is the name of calibration files required for the data analysis.
All the calibration files should be added to ``SCALES-DRP/scalesdrp/calib`` directory.

Bad pixel matrices (bpmats) are not configured: the pipeline builds one from
the bad pixel mask (and, for IFS science frames, the rectification matrix of
the frame's mode) the first time it is needed and caches it. See
:ref:`caldbconfig`.

.. code-block:: bash

  sig_map_ifs_fast0p6 = 'readnoise_ifs_fast0.6_cd5.fits' #readnoise map for IFS Fast0.6
  flat_ifs_fast0p6 = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Fast0.6
  bpm_ifs_fast0p6 = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Fast0.6
  lin_coeff_ifs_fast0p6 = 'lin_coeffs_ifs_fast0.6_cd5.fits' #linearity coefficients for IFS Fast0.6
  sat_map_ifs_fast0p6 = 'ifs_fast0.6_sat_map.fits'

  sig_map_ifs_fast1 = 'readnoise_ifs_fast1.0_cd5.fits' #readnoise map for IFS Fast1.0
  flat_ifs_fast1 = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Fast1.0
  bpm_ifs_fast1 = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Fast1.0
  lin_coeff_ifs_fast1 = 'lin_coeffs_ifs_fast1.0_cd5.fits' #linearity coefficients for IFS Fast1.0
  sat_map_ifs_fast1 = 'ifs_fast1.0_sat_map.fits'

  sig_map_ifs_slow = 'readnoise_ifs_slow_cd5.fits' #readnoise map for IFS Slow5.2
  flat_ifs_slow = 'ifs_fast0.6_pseudoflat_bpcorr.fits' # default pseudoflat for IFS Slow5.2
  bpm_ifs_slow = 'bpm_ifs_cd5.fits' #bad pixel mask for IFS Slow5.2
  lin_coeff_ifs_slow = 'lin_coeffs_ifs_slow_cd5.fits' #linearity coefficients for IFS Slow5.2
  sat_map_ifs_slow = 'ifs_slow_sat_map.fits'

  sig_map_img_fast0p6 = 'readnoise_img_fast0.6_cd5.fits' #readnoise map for IMG Fast0.6
  flat_img_fast0p6 = 'img_fast0.6_mflatlamp.fits' # default pseudoflat for IMG Fast0.6
  bpm_img_fast0p6 = 'bpm_img_cd4.fits' #bad pixel mask for IMG Fast0.6
  lin_coeff_img_fast0p6 = 'lin_coeffs_img_fast0.6_cd5.fits' #linearity coefficients for IMG Fast0.6
  sat_map_img_fast0p6 = 'img_fast0.6_sat_map.fits'

  sig_map_img_fast1 = 'readnoise_img_fast1.0_cd5.fits' #readnoise map for IMG Fast1.0
  flat_img_fast1 = 'img_fast1.0_mflatlamp.fits' # default pseudoflat for IMG Fast1.0
  bpm_img_fast1 = 'bpm_img_cd4.fits' #bad pixel mask for IMG Fast1.0
  lin_coeff_img_fast1 = 'lin_coeffs_img_fast1.0_cd5.fits' #linearity coefficients for IMG Fast1.0
  sat_map_img_fast1 = 'img_fast1.0_sat_map.fits'

  sig_map_img_slow = 'readnoise_img_slow_cd5.fits' #readnoise map for IMG Slow5.2
  flat_img_slow = 'img_slow_mflatlamp.fits' # default pseudoflat for IMG Slow5.2
  bpm_img_slow = 'bpm_img_cd4.fits' #bad pixel mask for IMG Slow5.2
  lin_coeff_img_slow = 'lin_coeffs_img_slow_cd5.fits' #linearity coefficients for IMG Slow5.2
  sat_map_img_slow = 'img_slow_sat_map.fits'

Calibration Module Specific Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have configuration parameters specific to calibration module. They are listed below:

.. code-block:: bash

	skip_mcal_generation = True ##Skip the generation of combined master cal files & go straight to spectral extraction analysis
	rectmat_xshift = 4.0 #Generate a second set of rectification matrices while #applying an x and/or y shift to compensate for disperser motion?
	rectmat_yshift = 0


.. _caldbconfig:

Calibration Database
~~~~~~~~~~~~~~~~~~~~

With ``use_caldb = True``, rectification matrices and bad pixel masks are
looked up in the calibration database (caldb) for every frame: the newest
``GOOD`` file of the right type, IFS mode and read mode taken no later than
the frame. Downloaded files are verified against their MD5 and kept in a
local cache, so the caldb is only contacted again when its contents change.
If the caldb has nothing suitable, the file named in the configuration is
used from the calib directory.

.. code-block:: bash

  use_caldb = False           # look up calibration files in the caldb
  caldb_url = ''              # e.g. 'http://caldb-host:5000'
  caldb_cache_dir = ''        # '' = $SCALES_CALIB_CACHE or ~/.cache/scalesdrp
  caldb_offline = False       # True = only use files already in the cache
  caldb_ttl = 3600            # seconds between checks for caldb updates
  caldb_match_stateid = False # True = only accept files with the frame's STATEID

The observer id sent to the caldb is read from ``$SCALES_CALDB_OBSID``. Bad
pixel matrices built by the pipeline are cached under the same directory.

The ``scales_caldb`` command lists what the caldb holds (``list``),
pre-fetches the current files for observing (``sync --bpm --bpmats``),
registers new calibration files using their FITS headers (``publish``) and
converts legacy ``.npz`` rectification matrices to FITS (``convert``).

Science-grade Module Specific Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have default parameters for science-grade module listed below.
The name of the default rectification matrix associated with each IFS module.
These files should be present in the ``SCALES-DRP/scalesdrp/calib`` directory;
they are used when the caldb is off or has no rectification matrix for the
frame. Both FITS and legacy ``.npz`` rectification matrices are read.

.. code-block:: bash

  OPT_rmat_LowRes_K = 'LowRes-K_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_K = 'LowRes-K_C2_intp_rectmat.npz'
  #OPT_rmat_LowRes_L = 'LowRes-L_OPT_intp_rectmat.npz'
  #C2_rmat_LowRes_L = 'LowRes-L_C2_intp_rectmat.npz'
  OPT_rmat_LowRes_L = 'LowRes-L_OPT_rectmat.npz'
  C2_rmat_LowRes_L = 'LowRes-L_C2_rectmat.npz'
  OPT_rmat_LowRes_M = 'LowRes-M_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_M = 'LowRes-M_C2_intp_rectmat.npz'
  OPT_rmat_LowRes_KLM = 'LowRes-KLM_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_KLM = 'LowRes-KLM_C2_intp_rectmat.npz'
  OPT_rmat_LowRes_KL = 'LowRes-KL_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_KL = 'LowRes-KL_C2_intp_rectmat.npz'
  OPT_rmat_LowRes_Ls = 'LowRes-Ls_OPT_intp_rectmat.npz'
  C2_rmat_LowRes_Ls = 'LowRes-Ls_C2_intp_rectmat.npz'
  OPT_rmat_MedRes_K = 'MedRes-K_OPT_intp_rectmat_260604.npz'
  #OPT_rmat_MedRes_K = 'MedRes-K_OPT_intp_rectmat_dx4.0_dy0.npz'
  C2_rmat_MedRes_K = 'MedRes-K_C2_intp_rectmat_260604.npz'
  #C2_rmat_MedRes_K = 'MedRes-K_C2_intp_rectmat_dx4.0_dy0.npz'
  OPT_rmat_MedRes_L = 'MedRes-L_OPT_intp_rectmat.npz'
  C2_rmat_MedRes_L = 'MedRes-L_C2_intp_rectmat.npz'
  OPT_rmat_MedRes_M = 'MedRes-M_OPT_intp_rectmat.npz'
  C2_rmat_MedRes_M = 'MedRes-M_C2_intp_rectmat.npz'

  do_chi2_full = False #enable chi square spectral extraction
  apply_lens_flat=False # apply lenslet flat correction
  # The cube shape and wavelength of each slice come from the rectification
  # matrix. These lenslet grids are only used for legacy .npz matrices that
  # do not store their cube shape.
  lowres_lenslet_grid = (112, 112)
  medres_lenslet_grid = (17, 18)
