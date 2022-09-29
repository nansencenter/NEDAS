.SUFFIXES: .o .F90 .

CF90 = mpiifort

#Optimalization opts
OPTO= -g -fpic -convert big_endian -assume byterecl -cm -w -O2 -r8 -traceback

F77FLG = -nofree
F90FLG = -free
CPPFLAGS += $(CPPARCH) -DFFTW -DIARGC -DLAPACK # -DQMPI
FLIBS = -L$(NETCDF)/lib -lnetcdff -L$(FFTW_DIR)/lib -lfftw3
INCLUDEDIR= -I$(NETCDF)/inlude -I$(FFTW_DIR)/include

FFLAGS = $(OPTO) $(INCLUDEDIR)

ifneq (,$(findstring -DAIX,$(CPPFLAGS)))
   subs=-WF,-
   CPPFLAGS:=$(subst -,$(subs),$(CPPFLAGS))
endif

# Rules for running cpp and updating files in include directory
.F90.o:
	$(CF90) -c $(CPPFLAGS) $(FFLAGS) $(F90FLG) -o $*.o $<

.F.o:
	$(CF77) -c $(CPPFLAGS) $(FFLAGS) $(F77FLG) -o $*.o $<

PEXEC = perturbation.exe

all  : $(PEXEC)

OBJS= \
	module_pseudo2d.o \
	module_random_field.o \
	main_perturbation.o

$(PEXEC): $(OBJS)
	$(CF90) $(OBJS) $(FLIBS) -o $(PEXEC)
#############################################################################

bin  : $(PEXEC)

clean:
	rm -f *.o *.mod
	rm -f $(PEXEC)
