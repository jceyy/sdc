This is a readme.


List of compile flags
Flag                            Defined by      Default Description
PERIODIC                        Makefile        -       Enable periodic boundaries in particle tracer (and possibly elsewhere)
PENALIZED                       Makefile        -       Enable penalization scheme for forbidden areas

WRITE_VTK                       cuda_defines.h  off     Write vtk-readable files of the velocity field in main.cu
WRITE_SNAPSHOT                  cuda_defines.h  off     Accept commandline option snapshot_freq to write gnuplot midplane temperature layer

CALC_STRUCTFN                   Makefile        off     Calculate the structure function of the flow field
CALC_ENERGYFN                   Makefile        off     Calculate the energy spectrum of the flow field

USE_PARTICLES                   Makefile        -       Use particle tracer
WRITE_TRACKS                    Makefile        -       Write particle tracks to binary .trk file

PRECISION_DOUBLE                Makefile        off     Do calculations in double precision

List of command line options
-device <int> (optional)
-dir <string> (optional)
-nmodes <int int int> (mandatory)
-clength <float float> (mandatory)

Init:
-init_rectangle (bool)
-init_dislocation (bool)
-init_isr (bool)
-init_sv (bool)
-init_cr (bool)
-init_giantspiral (bool)
-init_test (bool)
-init_mode <float> (optional)

Runtime vars:
-ra <float> (mandatory)
-pr <float> (mandatory)
-delta_t <float> (mandatory)
-final_t <float> (mandatory)
-nfiles <int> (optional)
-eta <float> (optional, ifdef PENALIZED)
-files <string string string string string> (optional)
-seed <int> (optional)
-mean_F <float> (optional)

Server:
-port <int> (optional, ifdef USE_SERVER)

Particles:
-npart <int> (optional)
-part_center (bool)
-part_center_sigma (bool)
-part_diagonal (bool)
-part_pairs (bool)
-part_coord_sigma <float float> (optional)
-part_sigma <float> (optional)

-write_tracks_every <int> (optional)
-calc_secondmoment (bool)
-le_count <int> (optional)
-le <float float ...> (optional)

-snapshot_freq <int> (optional)
-no_step_time (bool)
-freeze_circle <int> (optional)
