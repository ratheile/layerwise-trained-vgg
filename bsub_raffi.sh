#!/bin/bash
cd $HOME/scaling-spoon
bsub -n 4 -R "rusage[mem=16000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" ./run_raffi.sh "$1"
