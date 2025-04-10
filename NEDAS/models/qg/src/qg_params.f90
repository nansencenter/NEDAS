module qg_params                   !-*-f90-*-

  !************************************************************************
  ! Contains all global parameters for program, and routines for
  ! processing their I/O
  !
  ! Routines: Initialize_parameters, Write_parameters, Check_parameters
  !
  ! Dependencies: IO_tools, Syscalls
  !************************************************************************

  implicit none
  public
  save

  integer,parameter       :: ci=-9      ! For checking init status of variables
  real,parameter          :: cr=-9.

  ! Parameters in namelist input -- initialize some parameterss to
  ! negative values (cr and ci) in order to check that they are intentionally
  ! initialized in Parameters_ok (below)

  ! Resolution

  integer                 :: nz          = ci          ! Vertical resolution
  integer                 :: kmax        = ci          ! Spectral resolution
  real                    :: dt          = cr          ! Time step

  ! Flags

  logical                 :: restarting       = .false.! Is this a restart?
  logical                 :: use_tracer       = .false.! Calc tracer eqn
  logical                 :: use_topo         = .false.! Include topography
  logical                 :: adapt_dt         = .true. ! Adaptive timestep
  logical                 :: use_forcing      = .false.! Use rndm markov frcing
  logical                 :: norm_forcing     = .false.! Norm gen rate from RMF
  logical                 :: use_forcing_t    = .false.! Use RMF for tracers
  logical                 :: norm_forcing_t   = .false.! Norm gen rate trc RMF
  logical                 :: use_mean_grad_t  = .false.! Use mean trcr gradient
  logical                 :: do_spectra       = .true. ! Calc/save spectra
  logical                 :: do_aniso_spectra = .false.! Anisotropic spectra
  logical                 :: do_xfer_spectra  = .false.! Calc transfer spectra
  logical                 :: do_genm_spectra  = .false.! Calc modal generation
  logical                 :: do_x_avgs        = .false.! Calc zonal averages
  logical                 :: calc_residual    = .false.! Calc residual delta E
  logical                 :: linear           = .false.! Omit non-linear terms
  logical                 :: initialize_energy= .true. ! Set init energy to e_o
  logical                 :: limit_bot_drag   = .false.! Apply drg to k<kf_min
  logical                 :: read_tripint     = .false. ! Read in trip int coefs.

  ! Switches

  character(20)           :: psi_init_type  = ''       ! Init streamfn type
  character(20)           :: surface_bc     = ''       ! Surface BC type
  character(20)           :: strat_type     = ''       ! Stratification type
  character(20)           :: ubar_type      = ''       ! Mean u type
  character(20)           :: vbar_type      = ''       ! Mean v type
  character(20)           :: topo_type      = ''       ! Topography type
  character(20)           :: tracer_init_type = ''     ! Initial tracer type
  character(20)           :: filter_type    = ''       ! Spatial filter 
  character(20)           :: filter_type_t  = ''       ! Spatial filter tracer

  ! Fundamental scales

  real                    :: beta        = cr          ! beta_0*L^2/[(2pi)^2 U]
  real                    :: F           = cr          ! f^2*L^2/[(2pi)^2g'H_0]
  real                    :: uscale      = 0.          ! Scale factor for Ubar
  real                    :: vscale      = 0.          ! Scale factor for Vbar

  ! Mean stratification/velocity parameters

  real                    :: deltc       = cr          ! ~thermocline thickness
  real                    :: delu        = cr          ! ~Ubar surf intens'n
  integer                 :: umode       = ci          ! Mode of Ubar ('mode')
  real                    :: Fe          = 0           ! F*drho(1)/drho_ext
  
  ! Streamfunction initialization parameters (dep't on psi_init_type)

  real                    :: e_o         = cr          ! Initial energy
  real                    :: k_o         = cr          ! Initial peak k
  real                    :: delk        = cr          ! Initial spread in k
  real                    :: aspect_vort = cr          ! Initl vort aspct ratio
  real                    :: del_vort    = cr          ! Initial vortex width
  integer                 :: m_o         = ci          ! Initial modal peak
  integer                 :: z_o         = ci          ! Initial level

  ! Dissipation parameters

  real                    :: filter_exp  = cr          ! Filter exponent
  real                    :: k_cut       = cr          ! Exp cutoff scale
  real                    :: bot_drag    = 0.          ! Bottom drag
  real                    :: quad_drag   = 0.          ! Quadratic bottom drag
  real                    :: qd_angle    = 0.          ! Quad drag turn angle
  real                    :: top_drag    = 0.          ! Top drag
  real                    :: therm_drag  = 0.          ! Thermal drag
  real                    :: filt_tune   = 1.           ! Tuning for ens filter

  ! Markovian forcing parameters

  real                    :: forc_coef   = cr          ! BT forcing coefficient
  real                    :: forc_corr   = cr          ! BT forcing correlation
  real                    :: kf_min      = cr          ! min k^2 for BT frc
  real                    :: kf_max      = cr          ! max k^2 for BT frc

  ! Topography parameters

  real                    :: toposcale   = cr          ! Scale factor for topo
  real                    :: del_topo    = cr          ! Bump width in k or x
  real                    :: k_o_topo    = cr          ! Peak posn for kspc hb

  ! Tracer parameters

  real                    :: tvar_o      = cr          ! Init tracer variance
  real                    :: kf_min_t    = cr          ! min k^2 for trc frc
  real                    :: kf_max_t    = cr          ! max k^2 for trc frc
  real                    :: forc_coef_t = cr          ! Tracer forcing coef
  real                    :: k_cut_t     = cr          ! Exp cutoff wavenumber
  real                    :: filter_exp_t= cr          ! Filter_t exponent
  integer                 :: z_stir      = 1           ! psi level to stir trcr
  real                    :: filt_tune_t = 1.           ! Tuning for tvar filt

  ! Interval steps for i/o and diagnostic calculations

  integer                 :: write_step  = ci          ! Frame snapshot step
  integer                 :: diag1_step  = ci          ! Diagnostics 1 step
  integer                 :: diag2_step  = ci          ! Diagnostics 2 step

  ! Counters and start values

  integer                 :: total_counts= ci          ! Total timesteps to do
  integer                 :: start_frame = ci          ! Frame to start from
  integer                 :: cntr        = 1           ! Timestep counter value
  integer                 :: frame       = 0           ! Field snapshot frame
  integer                 :: d1frame     = 0           ! Diagnostics 1 frame
  integer                 :: d2frame     = 0           ! Diagnostics 2 frame
  real                    :: time        = 0.

  ! Input files

  character(70)           :: psi_init_file = ''        ! Psi input field
  character(70)           :: tracer_init_file = ''     ! Initial tracer field
  character(70)           :: ubar_in_file= ''          ! Ubar profile
  character(70)           :: vbar_in_file= ''          ! Vbar profile
  character(70)           :: dz_in_file  = ''          ! Layer thicknesses
  character(70)           :: rho_in_file = ''          ! Density profile
  character(70)           :: tripint_in_file = ''      ! Trip int coef input
  character(70)           :: hb_in_file  = ''          ! Bottom topo (spec)

  ! Date and time

  character(8)            :: end_date                  ! End date of sim
  character(10)           :: end_time                  ! End time of sim

  ! DIP Switches and tuning factors - factory settings.
  ! All of these are included in namelist input as well, but they
  ! are not checked for initialization in Parameters_ok so that you
  ! dont have to include them in the input file (but can if you need to).

  integer                 :: recunit     = 8           ! For direct access IO
  integer                 :: idum        = -7          ! Random generator seed

  ! Numerical stability tuning parameters

  real                    :: robert      = 0.01        ! Robert filter value
  real                    :: dt_tune     = 1.5         ! Tuning for adaptv dt
  integer                 :: dt_step     = 10          ! dt re-calc interval
  real                    :: rmf_norm_min= 1.e-5       ! RMF genn min 4 normn
  real                    :: rho_slope   = 5.e-5       ! Lin slope * exp prof
  integer                 :: hf          = 10          ! High res interp factr
                                                       !   for get_tripint
  real                    :: drt         = -9000.      ! 1st derivs at bnds for
  real                    :: drb         = -9000.      ! spline in get_tripint

  character(20)           :: dealiasing  = 'orszag'    ! Set de-aliasing form
  character(20)           :: dealiasing_t= 'orszag'    ! Set de-aliasing form

  ! **************End of namelist input parameters*************
  !
  ! Parameters for global internal use - NOT included in any namelist input
  !
  ! Output file and directory names - diagnostics outputs are defined in 
  ! respective diagnostics module.

  character(80)           :: datadir       = '.'
  character(32)           :: inputfile     = 'input.nml'
  character(32)           :: restartfile   = 'restart.nml'
  character(32)           :: psi_file      = 'psi'       ! Psi snapshots 
  character(32)           :: force_o_file  = 'force_o'   ! Forcing 
  character(32)           :: tracer_file   = 'tracer'    ! Tracer snapshots 
  character(32)           :: force_ot_file = 'force_ot'  ! Tracer forcing 
  character(32)           :: psiq_file     = 'psiq'
  character(32)           :: vmode_file    = 'vmode'
  character(32)           :: kz_file       = 'kz'
  character(32)           :: tripint_file  = 'tripint'
  character(32)           :: time_file     = 'write_time'
  character(50)           :: ubar_file     = 'ubar'      ! Ubar profile
  character(50)           :: vbar_file     = 'vbar'      ! Vbar profile
  character(50)           :: dz_file       = 'dz'        ! Layer thicknesses
  character(50)           :: rho_file      = 'rho'       ! Density profile
  character(50)           :: hb_file       = 'hb'        ! Bottom topo (spec)

  ! Resolution and tuning parameters (set as functions of kmax and nz) 

  integer                 :: numk, nkx, nky, nx, ny, nv, nmask

  ! Internal flags and counters

  logical                 :: surf_buoy  = .false.
  logical                 :: parameters_ok = .true.
  logical                 :: start
  integer                 :: cnt = 1, call_q = 0, call_b = 0, call_t = 0

  ! Cabalistic numbers 

  real,parameter          :: pi          = 3.14159265358979
  complex,parameter       :: i           = (0.,1.)

  ! Namelist declarations

  namelist/run_params/nz, kmax, dt
  namelist/run_params/restarting,adapt_dt,use_tracer,use_topo
  namelist/run_params/use_forcing,norm_forcing,use_forcing_t,norm_forcing_t
  namelist/run_params/use_mean_grad_t,linear,initialize_energy,read_tripint
  namelist/run_params/do_spectra,do_xfer_spectra,do_genm_spectra
  namelist/run_params/do_aniso_spectra,do_x_avgs,calc_residual
  namelist/run_params/psi_init_type,topo_type,surface_bc,tracer_init_type
  namelist/run_params/strat_type,ubar_type,vbar_type,filter_type,filter_type_t
  namelist/run_params/F,beta,uscale,vscale
  namelist/run_params/deltc,delu,umode,Fe
  namelist/run_params/k_o,delk,aspect_vort,del_vort,m_o,z_o,e_o
  namelist/run_params/filter_exp,k_cut,bot_drag,quad_drag,qd_angle
  namelist/run_params/therm_drag,top_drag
  namelist/run_params/forc_coef,forc_corr,kf_min,kf_max
  namelist/run_params/toposcale,del_topo,k_o_topo
  namelist/run_params/tvar_o,kf_min_t,kf_max_t,forc_coef_t,k_cut_t,z_stir
  namelist/run_params/filter_exp_t,write_step,diag1_step,diag2_step
  namelist/run_params/total_counts,start_frame,cntr,frame,d1frame,d2frame,time
  namelist/run_params/psi_init_file,ubar_in_file,vbar_in_file
  namelist/run_params/tripint_in_file,dz_in_file,rho_in_file
  namelist/run_params/tracer_init_file,hb_in_file
  namelist/run_params/end_date,end_time
  namelist/run_params/recunit,idum,dealiasing,dealiasing_t,filt_tune
  namelist/run_params/filt_tune_t,robert,dt_tune,rmf_norm_min,dt_step
  namelist/run_params/rho_slope,hf,drt,drb
  namelist/run_params/limit_bot_drag

  !************** Routines for parameter I/O*********************

contains

  subroutine Initialize_parameters

    !************************************************************************
    ! Read in command line arguments 'datadir' and 'inputfile'
    ! then read namelist in 'inputfile' for input parameters,
    ! and pass some parameters to 'io_tools' which are needed for I/O.
    ! If no first arg supplied, then datadir = ./, and if no
    ! 2nd arg, then inputfile = input.nml (in datadir)
    ! Any I/O errors in this part occur before program
    ! knows where to write 'run.log' to (I/O params not passed
    ! to io_mod yet), so errors are written to screen and to
    ! error.log in executable directory for non-interactive runs.
    !************************************************************************

    use io_tools, only: Message, Pass_params
    use syscalls, only: Get_arg

    character(80)         :: fnamein='', temp=''
    integer               :: fin=7, iock, nchars
    integer(kind=4)       :: nchars4
    logical               :: restart_exists=.false.

    call Get_arg(int(1,4),datadir,nchars4,iock)
    nchars = int(nchars4)
    if (iock/=0) call Message('1st getarg failed:'//trim(datadir),&
         iock,fatal='y')
    if (datadir(nchars:nchars)/='/') datadir=trim(datadir)//'/'
    call Pass_params(datadir,recunit) ! Send to io_mod

    ! If there is a restart file in the data directory, then use
    ! this for parameter input, unless there is a second cmnd line arg,
    ! which allows selection of input nml file.

    inquire(file=trim(datadir)//trim(restartfile),exist=restart_exists)
    if (restart_exists) inputfile = restartfile

    call Get_arg(int(2,4),temp,nchars4,iock)
    nchars = int(nchars4)
    if (iock/=0) then
       call Message('2nd getarg failed:'//trim(temp),tag=iock,fatal='y')
    else
       if (trim(temp)/='') then
          inputfile = temp
       endif
    endif

    fnamein = trim(datadir)//trim(inputfile)

    open(unit=fin,file=fnamein,status='old',delim="apostrophe",iostat=iock)
    if (iock/=0) call Message('Open input namelist error; file:'//fnamein, &
         tag=iock,fatal='y')
    read (unit=fin,nml=run_params,iostat=iock)
    if (iock/=0) call Message('Read input namelist error; file, iock ='&
         &//fnamein, tag=iock,fatal='y')
    close(fin)

    parameters_ok = Check_parameters()  ! See below - check and set counters

    nx = 2*(kmax+1)            ! Physical resolution in x
    ny = 2*(kmax+1)            ! Physical resolution in y
    nkx = 2*kmax+1             ! Number of zonal wavenumbers
    nky = kmax+1               ! Number of meridional wavenumbers
    numk = 2*kmax*(kmax+1)     ! Total number of (k,l) points in horiz plane
    idum = -abs(idum)          ! Make sure random num gen is set to start right
    nv = nz
    if (surf_buoy) nv = nz+1 ! Add surface layer to psi 

  end subroutine Initialize_parameters

  !**************************************************************

  subroutine Write_parameters

    !************************************************************************
    ! Write parameters to restart namelist file and selected
    ! params to a special file for use by SQG Matlab routines.
    !************************************************************************

    use io_tools, only: Message, Open_file

    character(70) :: restartname, exitcheckname
    integer       :: outf = 40,iock

    restartname = trim(datadir)//trim(restartfile)

    ! Set certain parameters the way they should be set for restarting,
    ! regardless of what goes on in the program.  Store temp vals as nec.

    restarting = .true.
    start_frame = frame
    call date_and_time(DATE=end_date,TIME=end_time)

    open (unit=outf,file=restartname,status='unknown',delim="apostrophe",&
          iostat=iock)
    if (iock /= 0) call Message('write params nml open error; iock =', &
                                 tag=iock,fatal='y')
    write (unit=outf,nml=run_params,iostat=iock)
    if (iock /= 0) call Message('write params nml write error; iock =', &
                                 tag=iock,fatal='y')
    close(outf)

    ! Write params to bin file for reading with MATLAB SQG function getparams.m

    call Open_file(outf,'parameters','unknown',9)
    write(unit=outf,rec=1) kmax,nz,F,beta,bot_drag,uscale,d1frame,d2frame,frame
    close(outf)

  end subroutine Write_parameters

  !**************************************************************

  logical function Check_parameters()

    !**************************************************************
    ! This function will test that certain input namelist params are consistent
    ! and print some messages with basic info for run
    !**************************************************************

    use io_tools, only: Message

    Check_parameters=.true.

    call Message('')
    call Message('QG model version 2.92')
    call Message('')
    call Message('Checking parameters for consistency')
    call Message('')
    call Message('Input file = '//trim(datadir)//inputfile)
    call Message('Output directory = '//trim(datadir))

    ! Resolution

    if (kmax==ci) then
       call Message('Error: kmax not set')
       Check_parameters=.false.
    elseif (mod(kmax+1,2)/=0) then
       call Message('Error: kmax must be odd - yours is: ',tag=kmax)
       Check_parameters=.false.
    else
       call Message('Horizontal spatial resolution =',tag=2*(kmax+1))
    endif

    if (nz==ci) then
       call Message('Error: nz not set')
       Check_parameters=.false.
    elseif (nz<1) then
       call Message('Error: nz must be positive - yours is:',tag=nz)
       Check_parameters=.false.
    else
       call Message('Vertical resolution =', tag=nz)
    endif

    if (adapt_dt) then
       call Message('Using adaptive time-stepping')
    else
       call Message('Using fixed time-step')
       if (dt==cr) then
          call Message('Error: dt must be set explicitly in non-adaptive mode')
          Check_parameters=.false.
       endif
    endif

    ! Check basic scales
    
    if (F==cr.and.(nz==1)) then
       call Message('Info: F not set - setting to 0')
       F = 0.
    elseif ((F<=0.).and.(nz>1)) then
       call Message('Error: require F>0 with nz>1')
       Check_parameters=.false.
    else
       call Message('Finite deformation radius run: F parameter =',r_tag=F)
    endif
    if (Fe/=0) then
       call Message('Free surface selected:  Fe=',r_tag=Fe)
       if (nz==2) call Message('For nz=2, need equal layer depths.  Note that tripint not calculated correctly for this case yet')
    endif
    if (therm_drag==0) then
       call Message('Thermal drag=0')
    else
       call Message('Thermal drag=',r_tag=therm_drag)
    endif
    if (beta==cr) then
       call Message('Error: beta not set')
       Check_parameters=.false.
    else
       call Message('Beta-plane: beta =',r_tag=beta)
    endif
    if (uscale==cr) then
       call Message('Info: uscale not set - setting to 0')
       uscale = 0.
    elseif (uscale/=cr) then
       call Message('Zonal mean shear: uscale =',r_tag=uscale)
    endif
    
    ! Check boundary conditions

    if (nz>1) then
       select case (trim(surface_bc))
       case ('rigid_lid')

          call Message('Rigid lid surface BC selected')

       case ('surf_buoy')

          call Message('Surface buoyancy advection ON')
          surf_buoy = .true.

       case ('periodic')

          call Message('Periodic vertical BC selected -- hyper-toroidal')
          if (trim(strat_type)/='linear') &
               call Message('Warning: Non-uniform stratification is not& 
               & well-posed with periodic vertical BC')

       case default

          call Message('Info:  No surface BC selected -- selecting rigid_lid &
               &by default.')
          call Message('Legal surface BC choices: &
               &rigid_lid|surf_buoy|periodic')
          surface_bc = 'rigid_lid'
          
       end select
    endif

    
  end function Check_parameters

!*************************************************************************

end module qg_params
