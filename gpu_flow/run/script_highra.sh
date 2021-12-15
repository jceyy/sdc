./rayleigh_periodic -device 0 -init_isr -dir /scratch.local/schuetz/video/highra1 -nmodes 256 4 10 -clength 4 0.1 -ra 100000 -pr 1 -eta 0.0 -dt 0.0001 -t 1 -nfiles 1000

./rayleigh_periodic -device 0 -init_isr -dir /scratch.local/schuetz/video/highra2 -nmodes 256 4 10 -clength 4 0.1 -ra 1000000 -pr 1 -eta 0.0 -dt 0.00001 -t 1 -nfiles 1000

#./extractvel -nmodes 128 4 10 -clength 2 0.5 -files ../data/highra1/theta_result.bin ../data/highra1/f_result.bin ../data/highra1/g_result.bin ../data/highra1/F_result.bin ../data/highra1/G_result.bin -write_vel_nz 100 -outfile_vel ../data/highra1/vel_result.txt -write_temp_nz 100 -outfile_temp ../data/highra1/temp_result.txt -select_dim 1 1
