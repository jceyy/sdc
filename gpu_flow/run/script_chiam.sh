#!/bin/bash

# Create data
iteration_list=(0 1 2 3 4 5 6 7 8 9)
ra_list=(2560 3074 3500 3930 4270)
le_list=("1e-0" "1e-1" "1e-2" "1e-3" "1e-4")

for i in "${iteration_list[@]}"
do
for ra in "${ra_list[@]}"
do
for le in "${le_list[@]}"
do
    ./rayleigh_circle -dir /scratch.local/schuetz/chiam/r${ra}_l${le}_${i} -nmodes 256 256 -clength 60 60 -ra ${ra} -pr 1 -dt 0.01 -t 10 -eta 0.02 -npart 1 -le 0 -seed 1${i} -nfiles 0
    ./rayleigh_circle -dir /scratch.local/schuetz/chiam/r${ra}_l${le}_${i} -files theta_result.bin f_result.bin g_result.bin F_result.bin G_result.bin -nmodes 256 256 -clength 60 60 -ra ${ra} -pr 1 -dt 0.01 -t 100 -eta 0.02 -npart 500 -le ${le} -nfiles 0

done
done
done