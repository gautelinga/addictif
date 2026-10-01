#!/usr/bin/env python

import sys, importlib

list_of_scripts = ["stokes",
                   "stokes_pressure",
                   "ade_steady",
                   "refine",
                   "postprocess_abc",
                   "postprocess_crn",
                   "postprocess_crn_kinetic",
                   "postprocess_crn_surface_reaction",
                   "postprocess_sr",
                   "postprocess_sr_kinetic",
                   "analyze_data",
                   "analyze_data_crn",
                   "compute_averages",
                   "compute_averages_abc",
                   "plot_scan",
                   "make_video",
                   "make_video_new"]

def main():
    assert len(sys.argv) > 1
    assert sys.argv[1] in list_of_scripts
    script = sys.argv.pop(1)
    m = importlib.import_module("addictif.scripts."+script)
    m.main()

if __name__ == "__main__":
    main()
