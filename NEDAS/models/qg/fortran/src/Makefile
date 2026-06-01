# Makefile created by mkmf $Id: mkmf,v 12.0 2005/04/14 19:59:08 fms Exp $ 

include mkmf.src

.DEFAULT:
	-touch $@
all: qg.exe
fft_fftw3_1pe.o: ./fft_fftw3_1pe.f90
	$(FC) $(FFLAGS) -c	./fft_fftw3_1pe.f90
qg_params.o: ./qg_params.f90 io_tools.o syscalls_linux.o
	$(FC) $(FFLAGS) -c	./qg_params.f90
qg_arrays.o: ./qg_arrays.f90 io_tools.o qg_params.o
	$(FC) $(FFLAGS) -c	./qg_arrays.f90
qg_run_tools.o: ./qg_run_tools.f90 op_rules.o qg_params.o qg_arrays.o numerics_lib.o io_tools.o
	$(FC) $(FFLAGS) -c	./qg_run_tools.f90
eig_pak.o: ./eig_pak.f90
	$(FC) $(FFLAGS) -c	./eig_pak.f90
qg_driver.o: ./qg_driver.f90 op_rules.o qg_arrays.o qg_params.o qg_run_tools.o qg_init_tools.o qg_diagnostics.o transform_tools.o numerics_lib.o io_tools.o
	$(FC) $(FFLAGS) -c	./qg_driver.f90
strat_tools.o: ./strat_tools.f90 io_tools.o numerics_lib.o eig_pak.o
	$(FC) $(FFLAGS) -c	./strat_tools.f90
qg_diagnostics.o: ./qg_diagnostics.f90 op_rules.o qg_params.o qg_arrays.o io_tools.o transform_tools.o numerics_lib.o strat_tools.o
	$(FC) $(FFLAGS) -c	./qg_diagnostics.f90
transform_tools.o: ./transform_tools.f90 fft_fftw3_1pe.o op_rules.o
	$(FC) $(FFLAGS) -c	./transform_tools.f90
op_rules.o: ./op_rules.f90 io_tools.o
	$(FC) $(FFLAGS) -c	./op_rules.f90
eigrg1.o: ./eigrg1.f90
	$(FC) $(FFLAGS) -c	./eigrg1.f90
syscalls_linux.o: ./syscalls_linux.f90
	$(FC) $(FFLAGS) -c	./syscalls_linux.f90
io_tools.o: ./io_tools.f90
	$(FC) $(FFLAGS) -c	./io_tools.f90
numerics_lib.o: ./numerics_lib.f90 op_rules.o io_tools.o
	$(FC) $(FFLAGS) -c	./numerics_lib.f90
qg_init_tools.o: ./qg_init_tools.f90 io_tools.o qg_params.o qg_arrays.o strat_tools.o transform_tools.o numerics_lib.o op_rules.o qg_diagnostics.o
	$(FC) $(FFLAGS) -c	./qg_init_tools.f90
SRC = ./fft_fftw3_1pe.f90 ./qg_params.f90 ./qg_arrays.f90 ./qg_run_tools.f90 ./eig_pak.f90 ./qg_driver.f90 ./strat_tools.f90 ./qg_diagnostics.f90 ./transform_tools.f90 ./op_rules.f90 ./eigrg1.f ./syscalls_linux.f90 ./io_tools.f90 ./numerics_lib.f90 ./qg_init_tools.f90
OBJ = fft_fftw3_1pe.o qg_params.o qg_arrays.o qg_run_tools.o eig_pak.o qg_driver.o strat_tools.o qg_diagnostics.o transform_tools.o op_rules.o eigrg1.o syscalls_linux.o io_tools.o numerics_lib.o qg_init_tools.o
clean: neat
	-rm -f .cppdefs $(OBJ) qg.exe
neat:
	-rm -f $(TMPFILES)
TAGS: $(SRC)
	etags $(SRC)
tags: $(SRC)
	ctags $(SRC)
qg.exe: $(OBJ)
	$(LD) $(OBJ) -o qg.exe $(LDFLAGS)
