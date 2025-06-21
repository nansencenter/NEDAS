# Makefile for DART modules
F90 = gfortran
FFLAGS = -O2 -ffree-line-length-none

MODULES = \
	types_mod.o \
	utilities_mod.o \
	random_seq_mod.o \
	sort_mod.o \
	time_manager_mod.o \
	null_mpi_utilities_mod.o \
	distribution_params_mod.o \
	normal_distribution_mod.o \
	gamma_distribution_mod.o \
	beta_distribution_mod.o \
	bnrh_distribution_mod.o \
	rootfinding_mod.o \
	kde_distribution_mod.o \
	probit_transform_mod.o

libdart.a: $(MODULES)
	ar rcs $@ $^

%.o: %.f90
	$(F90) $(FFLAGS) -c $<

clean:
	rm -f *.o *.mod *.a
