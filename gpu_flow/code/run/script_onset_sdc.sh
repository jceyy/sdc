#!/bin/bash

# Create initial fields
ra_list=(2900 3074 3500 4270)

for ra in "${ra_list[@]}"
do
./rayleigh_periodic -dir /scratch.local/schuetz/init_r${ra} -nmodes 256 256 -clength 100 100 -ra ${ra} -pr 1 -dt 0.01 -t 600 -eta 0.02 -seed ${ra} -device 1 -port 1235
done
