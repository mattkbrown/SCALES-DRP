"""
Reduce SCALES data with the Keck RTI pipeline and alert RTI (KOA real-time
ingestion) when each file is ready.

Based on kcwidrp/scripts/kcwi_rti.py and scalesdrp/scripts/reduce_scales.py
"""
from keckdrpframework.core.framework import Framework
from keckdrpframework.config.framework_config import ConfigClass
from keckdrpframework.utils.drpf_logger import getLogger
import datetime
import argparse
import sys
import traceback
import os
import shutil

from scalesdrp.pipelines.keck_rti_pipeline import Scales_pipeline as Keck_RTI_Pipeline
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
import logging.config


def _parse_arguments(in_args: list) -> argparse.Namespace:
    description = "SCALES RTI pipeline CLI"

    # this is a simple case where we provide a frame and a configuration file
    parser = argparse.ArgumentParser(prog=f"{in_args[0]}",
                                     description=description)
    parser.add_argument('-c', '--config', dest="SCALES_config_file", type=str,
                        help="SCALES configuration file", default=None)

    parser.add_argument('--rti-cfg', dest="rti_config_file", type=str,
                        help="RTI configuration file", default=None)
    parser.add_argument('--rti-ingesttype', choices=['lev1', 'lev2'],
                        dest="rti_ingesttype", type=str,
                        help="RTI ingest type", default=None, required=True)
    parser.add_argument('--write_config', dest="write_config",
                        help="Write out editable config files in current dir"
                        " (scales.cfg and rti.cfg)", action="store_true",
                        default=False)
    parser.add_argument('-f', '--frames', nargs='*', type=str,
                        help='input image files (full path, list ok)',
                        default=None)
    parser.add_argument('-l', '--list', dest='file_list',
                        help='File containing a list of files to be processed',
                        default=None)

    parser.add_argument('-i', '--infiles', dest="infiles",
                        help="Input files, or pattern to match", nargs="?")
    parser.add_argument('-d', '--directory', dest="dirname", type=str,
                        help="Input directory", nargs='?', default=None)
    # after ingesting the files,
    # do we want to continue monitoring the directory?
    parser.add_argument('-m', '--monitor', dest="monitor",
                        help='Continue monitoring the directory '
                             'after the initial ingestion',
                        action='store_true', default=False)

    # special arguments, ignore
    parser.add_argument("-I", "--ingest_data_only", dest="ingest_data_only",
                        action="store_true",
                        help="Ingest data and terminate")
    parser.add_argument("-w", "--wait_for_event", dest="wait_for_event",
                        action="store_true", help="Wait for events")
    parser.add_argument("-W", "--continue", dest="continuous",
                        action="store_true",
                        help="Continue processing, wait for ever")
    parser.add_argument("-s", "--start_queue_manager_only",
                        dest="queue_manager_only", action="store_true",
                        help="Starts queue manager only, no processing",)

    # scales specific parameters
    parser.add_argument("-p", "--proctab", dest='proctab', help='Proctab file',
                        default=None)

    out_args = parser.parse_args(in_args[1:])
    return out_args


def check_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print("Directory %s has been created" % directory)


def main():

    # Package
    pkg = 'scalesdrp'

    # get arguments
    args = _parse_arguments(sys.argv)

    if args.write_config:
        for cfg_name in ['scales.cfg', 'rti.cfg']:
            dest = os.path.join(os.getcwd(), cfg_name)
            if os.path.exists(dest):
                print(f"Config file {cfg_name} already exists in current dir")
            else:
                cfg_fullpath = get_resource_path(pkg, 'configs/' + cfg_name)
                shutil.copy(cfg_fullpath, os.getcwd())
                print(f"Copied {cfg_name} into current dir.  Edit and use "
                      "with -c / --rti-cfg")
        sys.exit(0)

    if args.file_list:
        if '.fits' in args.file_list:
            print("\nERROR - trying to read in fits file as file list\n\n"
                  "Please use -f or --frames for direct input of fits files\n")
            sys.exit(0)

    # START HANDLING OF CONFIGURATION FILES ##########

    # check for the logs diretory
    check_directory("logs")
    # check for the plots directory
    check_directory("plots")

    framework_config_file = "configs/framework.cfg"
    framework_config_fullpath = str(get_resource_path(pkg, framework_config_file))

    framework_logcfg_file = 'configs/logger.cfg'
    framework_logcfg_fullpath = str(get_resource_path(pkg, framework_logcfg_file))

    if args.SCALES_config_file is None:
        scales_config_file = 'configs/scales.cfg'
        scales_config_fullpath = str(get_resource_path(pkg, scales_config_file))
    else:
        scales_config_fullpath = os.path.abspath(args.SCALES_config_file)
    scales_config = ConfigClass(scales_config_fullpath, default_section='SCALES')

    if args.rti_config_file is None:
        rti_config_file = "configs/rti.cfg"
        rti_config_fullpath = str(get_resource_path(pkg, rti_config_file))
    else:
        rti_config_fullpath = os.path.abspath(args.rti_config_file)
    rti_config = ConfigClass(rti_config_fullpath, default_section='RTI')
    # END HANDLING OF CONFIGURATION FILES ##########

    # Add current working directory to config info
    scales_config.cwd = os.getcwd()

    # check for the output directory
    check_directory(scales_config.output_directory)

    try:
        framework = Framework(Keck_RTI_Pipeline, framework_config_fullpath)
        # add this line ONLY if you are using a local logging config file
        logging.config.fileConfig(framework_logcfg_fullpath)
        framework.config.instrument = scales_config
        framework.config.rti = rti_config
        framework.config.rti.rti_ingesttype = args.rti_ingesttype
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)
    framework.context.pipeline_logger = getLogger(framework_logcfg_fullpath,
                                                  name="SCALES")
    framework.logger = getLogger(framework_logcfg_fullpath, name="DRPF")

    if args.infiles is not None:
        framework.config.file_type = args.infiles

    # update proc table argument
    if args.proctab:
        framework.context.pipeline_logger.info(
            "Using proc table file %s" % args.proctab
        )
        framework.config.instrument.procfile = args.proctab

    # initialize the proctab and read it
    framework.context.proctab = Proctab()
    framework.context.proctab.read_proctab(framework.config.instrument.procfile)

    # calibration files and processing options used by the primitives
    # (kept identical to reduce_scales.py)
    framework.context.clobber = scales_config.clobber
    framework.context.calib_file_path = scales_config.calib_file_path

    framework.context.bpm_ifs_fast0p6 = scales_config.bpm_ifs_fast0p6
    framework.context.bpmat_ifs_fast0p6 = scales_config.bpmat_ifs_fast0p6
    framework.context.flat_ifs_fast0p6 = scales_config.flat_ifs_fast0p6
    framework.context.sig_map_ifs_fast0p6 = scales_config.sig_map_ifs_fast0p6
    framework.context.lin_coeff_ifs_fast0p6 = scales_config.lin_coeff_ifs_fast0p6
    framework.context.sat_map_ifs_fast0p6 = scales_config.sat_map_ifs_fast0p6
    
    framework.context.bpm_ifs_fast1 = scales_config.bpm_ifs_fast1
    framework.context.bpmat_ifs_fast1 = scales_config.bpmat_ifs_fast1
    framework.context.flat_ifs_fast1 = scales_config.flat_ifs_fast1
    framework.context.sig_map_ifs_fast1 = scales_config.sig_map_ifs_fast1
    framework.context.lin_coeff_ifs_fast1 = scales_config.lin_coeff_ifs_fast1
    framework.context.sat_map_ifs_fast1 = scales_config.sat_map_ifs_fast1

    framework.context.bpm_ifs_slow = scales_config.bpm_ifs_slow
    framework.context.bpmat_ifs_slow = scales_config.bpmat_ifs_slow
    framework.context.flat_ifs_slow = scales_config.flat_ifs_slow
    framework.context.sig_map_ifs_slow = scales_config.sig_map_ifs_slow
    framework.context.lin_coeff_ifs_slow = scales_config.lin_coeff_ifs_slow
    framework.context.sat_map_ifs_slow = scales_config.sat_map_ifs_slow

    framework.context.bpm_img_fast0p6 = scales_config.bpm_img_fast0p6
    framework.context.bpmat_img_fast0p6 = scales_config.bpmat_img_fast0p6
    framework.context.flat_img_fast0p6 = scales_config.flat_img_fast0p6
    framework.context.sig_map_img_fast0p6 = scales_config.sig_map_img_fast0p6
    framework.context.lin_coeff_img_fast0p6 = scales_config.lin_coeff_img_fast0p6
    framework.context.sat_map_img_fast0p6 = scales_config.sat_map_img_fast0p6

    framework.context.bpm_img_fast1 = scales_config.bpm_img_fast1
    framework.context.bpmat_img_fast1 = scales_config.bpmat_img_fast1
    framework.context.flat_img_fast1 = scales_config.flat_img_fast1
    framework.context.sig_map_img_fast1 = scales_config.sig_map_img_fast1
    framework.context.lin_coeff_img_fast1 = scales_config.lin_coeff_img_fast1
    framework.context.sat_map_img_fast1 = scales_config.sat_map_img_fast1

    framework.context.bpm_img_slow = scales_config.bpm_img_slow
    framework.context.bpmat_img_slow = scales_config.bpmat_img_slow
    framework.context.flat_img_slow = scales_config.flat_img_slow
    framework.context.sig_map_img_slow = scales_config.sig_map_img_slow
    framework.context.lin_coeff_img_slow = scales_config.lin_coeff_img_slow
    framework.context.sat_map_img_slow = scales_config.sat_map_img_slow

    framework.context.OPT_rmat_LowRes_K = scales_config.OPT_rmat_LowRes_K
    framework.context.C2_rmat_LowRes_K = scales_config.C2_rmat_LowRes_K
    framework.context.OPT_rmat_LowRes_L = scales_config.OPT_rmat_LowRes_L
    framework.context.C2_rmat_LowRes_L = scales_config.C2_rmat_LowRes_L
    framework.context.OPT_rmat_LowRes_M = scales_config.OPT_rmat_LowRes_M
    framework.context.C2_rmat_LowRes_M = scales_config.C2_rmat_LowRes_M
    framework.context.OPT_rmat_LowRes_KLM = scales_config.OPT_rmat_LowRes_KLM
    framework.context.C2_rmat_LowRes_KLM = scales_config.C2_rmat_LowRes_KLM
    framework.context.OPT_rmat_LowRes_KL = scales_config.OPT_rmat_LowRes_KLM
    framework.context.C2_rmat_LowRes_KL = scales_config.C2_rmat_LowRes_KLM
    framework.context.OPT_rmat_LowRes_Ls = scales_config.OPT_rmat_LowRes_KLM
    framework.context.C2_rmat_LowRes_Ls = scales_config.C2_rmat_LowRes_KLM
    framework.context.OPT_rmat_MedRes_K = scales_config.OPT_rmat_MedRes_K
    framework.context.C2_rmat_MedRes_K = scales_config.C2_rmat_MedRes_K
    framework.context.OPT_rmat_MedRes_L = scales_config.OPT_rmat_MedRes_L
    framework.context.C2_rmat_MedRes_L = scales_config.C2_rmat_MedRes_L
    framework.context.OPT_rmat_MedRes_M = scales_config.OPT_rmat_MedRes_M
    framework.context.C2_rmat_MedRes_M = scales_config.C2_rmat_MedRes_M

    framework.context.subtract_row_median = scales_config.subtract_row_median
    framework.context.do_swap = scales_config.do_swap
    framework.context.nchans = scales_config.nchans
    framework.context.altcol = scales_config.altcol
    framework.context.channelwise = scales_config.channelwise
    framework.context.amp_mean_func = scales_config.amp_mean_func
    framework.context.do_acn = scales_config.do_acn
    framework.context.acn_avg_type = scales_config.acn_avg_type
    framework.context.acn_mean_func = scales_config.acn_mean_func
    framework.context.acn_smooth = scales_config.acn_smooth
    framework.context.acn_savgol = scales_config.acn_savgol
    framework.context.acn_winsize = scales_config.acn_winsize
    framework.context.acn_order = scales_config.acn_order
    framework.context.resid_colsub = scales_config.resid_colsub
    framework.context.fixcol = scales_config.fixcol
    framework.context.ref_avg_type = scales_config.ref_avg_type
    framework.context.ref_mean_func = scales_config.ref_mean_func
    framework.context.ref_smooth = scales_config.ref_smooth
    framework.context.ref_savgol = scales_config.ref_savgol
    framework.context.ref_winsize = scales_config.ref_winsize
    framework.context.ref_order = scales_config.ref_order
    framework.context.pickup = scales_config.pickup
    framework.context.sigma_thresh = scales_config.sigma_thresh
    framework.context.dilate_iter = scales_config.dilate_iter
    framework.context.highpass_size = scales_config.highpass_size
    framework.context.per_amp = scales_config.per_amp
    framework.context.do_linearity = scales_config.do_linearity
    framework.context.apply_sat_mask = scales_config.apply_sat_mask
    framework.context.apply_bpm = scales_config.apply_bpm
    framework.context.apply_dark = scales_config.apply_dark
    framework.context.apply_det_flat = scales_config.apply_det_flat
    framework.context.apply_bias = scales_config.apply_bias
    framework.context.apply_lens_flat = scales_config.apply_lens_flat
    framework.context.lowres_final_cube = scales_config.lowres_final_cube
    framework.context.medres_final_cube = scales_config.medres_final_cube

    framework.logger.info("Framework initialized")
    framework.logger.info(f"RTI url is {framework.config.rti.rti_url}")

    # start queue manager only (useful for RPC)
    if args.queue_manager_only:
        # The queue manager runs forever.
        framework.logger.info("Starting queue manager only, no processing")
        framework.start(args.queue_manager_only)

    # single frame processing
    elif args.frames:
        framework.ingest_data(None, args.frames, False)

    # processing of a list of files contained in a file
    elif args.file_list:
        frames = []
        with open(args.file_list) as file_list:
            for frame in file_list:
                if "#" not in frame:
                    frames.append(frame.strip('\n'))
        framework.ingest_data(None, frames, False)

        with open(args.file_list + '_ingest', 'w') as ingest_f:
            ingest_f.write('Files ingested at: ' +
                           datetime.datetime.now().isoformat())

    # ingest an entire directory, trigger "next_file" (which is an option
    # specified in the config file) on each file,
    # optionally continue to monitor if -m is specified
    elif args.dirname is not None:
        framework.ingest_data(args.dirname, None, args.monitor)

    framework.config.instrument.wait_for_event = args.wait_for_event
    framework.config.instrument.continuous = args.continuous

    framework.start(args.queue_manager_only, args.ingest_data_only,
                    args.wait_for_event, args.continuous)


if __name__ == "__main__":
    main()
