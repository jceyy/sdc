import sys
import os
import fnmatch
from os.path import join
progpath = '~/code/single/run'
datapath = '../data'

# Create tracks of particle blob at center from giant spiral run.
# Argument 1: Rayleigh number
# Argument 2: Device number
# Argument 3: Lewis number

# should be: Ra 2647 Le 1e-3

ra = sys.argv[1]
dev = sys.argv[2]
le = sys.argv[3]

if(1==1):
    job = 'r'+ra+'_giantspiralb_le'+le

    # Create initial flow field
    os.system(join(progpath,"rayleigh_circle") +' -device '+dev+' -dir '+join(datapath,job)+' -nmodes 512 512 2 -clength 70 70 -ra '+ra+' -init_giantspiral -freeze_circle 500 -pr 1 -dt 0.01 -t 1000 -eta 0.02 -nfiles 0 -init_mode 25')

    exit()
    # rename output to input for particle diffusing
    for t in ['theta', 'f', 'g', 'F', 'G']:
        os.rename(join(datapath,job,t+'_result.bin'), join(datapath,job,t+'_init.bin'))

    # Simulate
    os.system(join(progpath,"rayleigh_circle") +' -device '+dev+' -dir '+join(datapath,job)+' -files theta_init.bin f_init.bin g_init.bin F_init.bin G_init.bin -nmodes 512 512 2 -clength 113.6 113.6 -ra '+ra+' -pr 1 -dt 0.01 -t 1001 -eta 0.02 -npart 100000 -part_center_sigma -le '+le+' -no_step_time -write_tracks_every 100 -nfiles 0 -snapshot_freq 1000')
