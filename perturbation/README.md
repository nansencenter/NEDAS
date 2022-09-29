This offline perturbation code originially comes from `HYCOM/src/mod_random_forcing.F`. For more details about the original code contact Laurent Bertino laurent.bertino@nersc.no

Modified by Yue Ying 2022

How to use:

1. Set correct environment variables and modules: `source ../env/$machine/perturbation.src`

2. Compile the perturbation code: `make`

3. Go to runtime directory, `ln -fs $NEDAS_DIR/perturbation/perturbation.exe .` and make changes to namelist `pseudo2d.nml` accordingly, then run the program `./perturbation.exe`. There will be `synforc_*.nc` files generated. The file contains perturbations for different variables with dimension (xdim, ydim)

List of namelist options:
```
&pseudo2d
      ! debug = .false.    ! debug mode
!      randf        = .true.  ! Switches on/off random forcing
!      seed         = 11
!!!    variances of variables (std**2)
!      vars%slp     =  10.0
!      vars%taux    =  1.e-3
!      vars%tauy    =  1.e-3
!      vars%wndspd  =  0.64
!      vars%clouds  =  5.e-3
!      vars%airtmp  =  9.0
!      vars%precip  =  1.0    ！ =1.0 means relative errors of 100%.
!      vars%relhum  =  1.0    ！ =1.0 means relative errors of 100%.
!      rf_hradius   =  500    ! Horizontal decorr length for rand forc [km];
!      dx = ! grid resolution
!      rf_tradius   =  2.0    ! Temporal decorr length for rand forc
!      rf_prsflg    =  2      ! Pressure flag must be between 0 and 2
/
```



<!--- In ./src/main_pseudo2D.F90,     -->
<!--    - Set the length of a sequential perturbations for one member to variable i_step.-->
<!--    - Set domain size to xdim = 1024, ydim = 1024,xy_full = xdim*ydim.  It uses FFT, which generates faster when using power of 2.-->

<!--- THEN, compile the code by makefile in ./src.-->

<!--- Configuration of perturbations are set in pseudo2D.nml-->
<!--- set mod_random_forcing.F90/rdtime as time step of forcing update. Also check the consistency with tcorr in pseudo2D.nml.-->

<!--- In ./result folder, pertubation series are saved in subfolders distincted by ensemble id. For examples,-->
<!--    -mem1 containts perturbations in netcdf as synforc_i.nc-->
<!--    ncdump -h synforc_1.nc shows-->
<!--        netcdf synforc_1 {-->
<!--        dimensions:-->
<!--            xy = 1048576 ;       (! xy =xdim*ydim)-->
<!--        variables:               (! the following variables related variances are defined in pesudo.nml. uwind,vwind are horizontal wind speed in u,v directions. The other variables are independent (not correlated). One can add/reduce variables on the specific needs.)-->
<!--            float uwind(xy) ;    -->
<!--            float vwind(xy) ;-->
<!--            float snowfall(xy) ;-->
<!--            float Qlw_in(xy) ;-->
<!--            float sss(xy) ;-->
<!--            float sst(xy) ;-->
<!--        }-->
<!--- In ./report folder, it saves a document records previous studies. The estimation of the amplification is in ./report/get_ratio.m or .py-->

<!--Use run_script.sh for a fresh compilation and generating perturbations, where ensemble size is given. The code can run in sequential or parallel on HPC-->
