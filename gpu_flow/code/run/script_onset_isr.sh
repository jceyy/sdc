#!/bin/bash

# Test case

#./rayleigh_periodic -device 0  -nmodes 500 500 2 -clength 100 100 -ra 2650 -pr 0.96 -eta 0.01 -dt 0.02 -t 600 -nfiles 0 -init_mode 25. #-snapshot_freq 20 -dir testcase0
./rayleigh_periodic -nmodes 2048 2048 2 -clength 100 100 -ra 2650 -pr 0.96 -eta 0.01 -dt 0.02 -t 500 -dir testcase0 -nfiles 0 -init_mode .25
