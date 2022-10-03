This offline perturbation code originially comes from `HYCOM/src/mod_random_forcing.F`. For more details about the original code contact Laurent Bertino laurent.bertino@nersc.no

Cleaned up and modified by Yue Ying 2022

How to use:

1. Set correct environment variables and modules: `source ../env/$machine/perturbation.src`

2. Compile the perturbation code: `make`

3. Go to runtime directory, `ln -fs $NEDAS_DIR/perturbation/perturbation.exe .` and make changes to namelist `pseudo2d.nml` accordingly, then run the program `./perturbation.exe`. There will be `synforc_*.nc` files generated. The file contains perturbations for different variables with dimension (xdim, ydim)

List of namelist options:
```
&pseudo2d
debug = .true.   !!switch on/off debug mode

xdim = 800 !!num grid points in x
ydim = 600 !!num grid points in y
dx = 5     !!grid resolution (km)
dt = 6     !!time step interval (hours)

n_sample = 10   !!number of perturbations to generate in time
nens = 20     !!ensemble size
n_field = 3   !!number of variables to be perturbed (max 100)

field(1)%name = 'slp     '     !!variable name (len=8)
field(1)%vars = 10.            !!variance of perturbation (in their units)
field(1)%hradius = 250.        !!horizontal correlation length (km)
field(1)%tradius = 48.         !!time correlation length (hours)

field(2)%name = 'uwind   '     !!more variables can be added with field(i) for i=1,...,n_field
field(2)%vars = 3.
field(2)%hradius = 250.
field(2)%tradius = 48.

field(3)%name = 'vwind   '
field(3)%vars = 3.
field(3)%hradius = 250.
field(3)%tradius = 48.

field(4)%name = 'snowfall'
field(4)%vars = 1.
field(4)%hradius = 250.
field(4)%tradius = 48.

prsflg = 1    !!flag for computing wind perturbation: 0=uncorrelated with slp;
              !!  1=derived from slp perturbations (see module_random_field.F90 for details)
/
```
