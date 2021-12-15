###########################################################################
#some macros for compilation
##########################################################################

INCLUDES = /usr/local/cuda/include
LIB_PATH = /usr/local/cuda/lib64
LIBS = -lcudart -lcufft -lgsl -lgslcblas -lm 


#TARGET = rayleigh
OPATH = obj_path
CPATH = ../code

###########################################################################
#some additional macros
############################################################################
CC_OPTIONS = -arch=sm_61  -O3 -I ${INCLUDES} -L ${LIB_PATH} ${LIBS} -Xcompiler -O3 -Xcompiler -march=native -Xcompiler -Wextra -Xcompiler -fdiagnostics-show-option -Xcompiler -Wno-unused-function -Xcompiler -Wno-long-long
COMPILER = /usr/local/cuda/bin/nvcc
CCOMPILER = g++

periodic: COMPILE_OPTIONS = ${CC_OPTIONS} -DPERIODIC -DUSE_SERVER -DUSE_PARTICLES -DWRITE_TRACKS -DTIMER -std=c++11
periodic: TARGET = rayleigh_periodic
periodic: path
periodic: compile

diskarray: COMPILE_OPTIONS = ${CC_OPTIONS} -DPENALIZED -DDISKARRAY -DUSE_SERVER -DUSE_PARTICLES -DTIMER -std=c++11 
diskarray: TARGET = rayleigh_disk_array
diskarray: path
diskarray: compile

circle: COMPILE_OPTIONS = ${CC_OPTIONS} -DPENALIZED -DCIRCLE -DUSE_SERVER -DUSE_PARTICLES -DTIMER -std=c++11
circle: TARGET = rayleigh_circle
circle: path
circle: compile

# Old / Only to maintain compatibility
mperiodic: COMPILE_OPTIONS = ${CC_OPTIONS} -DPERIODIC -DUSE_SERVER -DWRITE_TRACKS -DUSE_PARTICLES -DMULTITRACER -DTIMER -DWRITE_SNAPSHOT
mperiodic: TARGET = rayleigh_mperiodic
mperiodic: path
mperiodic: compile

mcircle: COMPILE_OPTIONS = ${CC_OPTIONS} -DPENALIZED -DCIRCLE -DUSE_SERVER -DUSE_PARTICLES -DMULTITRACER -DCALC_SECONDMOMENT -DTIMER -DWRITE_SNAPSHOT
mcircle: TARGET = rayleigh_mcircle
mcircle: path
mcircle: compile

rectangle: COMPILE_OPTIONS = ${CC_OPTIONS} -DPENALIZED -DUSE_SERVER -DUSE_PARTICLES -DTIMER -DWRITE_SNAPSHOT
rectangle: TARGET = rayleigh_rectangle
rectangle: path
rectangle: compile

# Remove .o files
clean:
	-rm -r obj_*

# Create symbolic link to target-specific output directory
path:
	mkdir -p obj_${TARGET}
	-rm ${OPATH}
	ln -s obj_${TARGET} ${OPATH}

compile: ${OPATH}/main.o ${OPATH}/init.o ${OPATH}/util.o ${OPATH}/inputparser.o ${OPATH}/coeff.o ${OPATH}/matrix_device.o ${OPATH}/matrix_folder.o ${OPATH}/matrix_host_iterator.o ${OPATH}/matrix_host.o ${OPATH}/calculate_temperature_operator.o ${OPATH}/calculate_velocity_operator.o ${OPATH}/inverse_B_matrix.o ${OPATH}/linear_implicit_operator.o ${OPATH}/nonlinear_operator_rayleigh_noslip.o ${OPATH}/freeze_operator.o ${OPATH}/masktemp.o ${OPATH}/masking.o ${OPATH}/particle_tracer.o ${OPATH}/datapoint_writer.o ${OPATH}/track_writer.o ${OPATH}/matrix_folder_writer.o ${OPATH}/server_writer.o ${OPATH}/serverclient.o ${OPATH}/energy_spectrum_2d.o ${OPATH}/structure_function_2d.o ${OPATH}/implicit_explicit_stepping.o ${OPATH}/track_movie.o
	${COMPILER} -link ${OPATH}/*.o -o ${TARGET} ${COMPILE_OPTIONS}


convert: COMPILE_OPTIONS = ${CC_OPTIONS}
convert: TARGET = convert
convert: path
convert: convert2

convert2: ${OPATH}/main_convert.o ${OPATH}/matrix_device.o ${OPATH}/matrix_folder.o ${OPATH}/matrix_host_iterator.o ${OPATH}/matrix_host.o ${OPATH}/matrix_folder_writer.o
	${COMPILER} -link ${OPATH}/*.o -o ${TARGET} ${COMPILE_OPTIONS}

distribution: COMPILE_OPTIONS = ${CC_OPTIONS}
distribution: TARGET = distribution
distribution: path
distribution: distribution2

distribution2: ${OPATH}/main_distribution.o ${OPATH}/inputparser.o ${OPATH}/track_movie.o
	${COMPILER} -link ${OPATH}/*.o -o ${TARGET} ${COMPILE_OPTIONS}

dispersion: COMPILE_OPTIONS = ${CC_OPTIONS}
dispersion: TARGET = dispersion
dispersion: path
dispersion: dispersion2

dispersion2: ${OPATH}/main_dispersion.o ${OPATH}/inputparser.o ${OPATH}/track_movie.o
	${COMPILER} -link ${OPATH}/*.o -o ${TARGET} ${COMPILE_OPTIONS}

extractvel: COMPILE_OPTIONS = ${CC_OPTIONS}
extractvel: TARGET = extractvel
extractvel: path
extractvel: extractvel2

extractvel2: ${OPATH}/main_extract_vel.o ${OPATH}/util.o ${OPATH}/inputparser.o ${OPATH}/coeff.o ${OPATH}/matrix_device.o ${OPATH}/matrix_folder.o ${OPATH}/matrix_host_iterator.o ${OPATH}/matrix_host.o ${OPATH}/calculate_temperature_operator.o ${OPATH}/calculate_velocity_operator.o  ${OPATH}/matrix_folder_writer.o ${OPATH}/datapoint_writer.o
	${COMPILER} -link ${OPATH}/*.o -o ${TARGET} ${COMPILE_OPTIONS}

${OPATH}/%.o: ${CPATH}/%.c*
	${COMPILER} -c $< -o $@ ${COMPILE_OPTIONS}

${OPATH}/%.o: ${CPATH}/*/%.c* ${CPATH}/*/%.h
	${COMPILER} -c $< -o $@ ${COMPILE_OPTIONS}

