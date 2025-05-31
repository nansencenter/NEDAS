module qg_init_tools         !-*-f90-*-  <- tells emacs: use f90-mode

  !************************************************************************
  ! Contains all the routines for initializing fields
  ! IO done here is for reading input fields, error msgs, and writing
  ! preset fields.
  !
  ! Routines:  Init_strat, Init_ubar, Init_topo, Init_tracer, 
  !            Init_streamfunction, Init_filter, Normu, Init_counters
  !
  ! Dependencies:  qg_transform_tools, io_tools, qg_arrays, qg_params,
  !                qg_strat_tools,qg_run_tools, op_rules, numerics_lib,
  !                qg_diagnostics
  !************************************************************************

  implicit none
  private
  save

  public :: Init_strat, Init_ubar, Init_topo, Init_tracer, &
            Init_streamfunction, Init_filter, Init_counters, &
            Init_forcing

contains

  !************************************************************************

  subroutine Init_counters

    !************************************************************************
    ! Initialize diagnostic and snapshot frame counters
    !************************************************************************

    use io_tools,  only: Message
    use qg_params, only: ci,cr,start_frame,frame,parameters_ok,cntr,cnt, &
                         d1frame,d2frame,write_step,diag1_step,diag2_step, &
                         total_counts,do_spectra,restarting
                         
    restart: if (restarting) then

       call Message('This is a restart run')
       if (start_frame==ci) then
          call Message('Error: start_frame must be set when restarting')
          parameters_ok = .false.
       elseif (start_frame<0.or.start_frame>frame) then
          call Message('Error: require 0 <= start_frame <= frame')
          parameters_ok = .false.
       elseif (start_frame<=frame) then      ! Set counters for restart
          frame = start_frame 
          if (frame==0) then
             cntr = 1
             d1frame = 0
             d2frame = 0
          else
             cntr = frame*write_step
             d1frame = (cntr-1-mod(cntr-1,diag1_step))/diag1_step + 1
             d2frame = (cntr-1-mod(cntr-1,diag2_step))/diag2_step + 1
          endif
       endif

    elseif (.not.restarting) then

       frame = 0; d1frame = 0; d2frame = 0; cntr = 1

       call Message('Checking counters...')  ! Check internal counters
       
       if (total_counts==ci) then
          call Message('Error: total_counts not set')
          parameters_ok=.false.
       else
          call Message('Total number of timesteps model will run =', &
                        tag=total_counts)
       endif
       if (diag1_step==ci) then
          call Message('Info: timeseries write interval not set - &
               &setting diag1_step = 50')
          diag1_step = 100
       endif
       call Message('Timeseries will be written at timestep interval =', &
                     tag=diag1_step)
       if (diag2_step==ci.and.do_spectra) then
          call Message('Info: spectra write interval not set - &
               &setting diag2_step = 100')
          diag2_step = 100
       endif
       call Message('Spectra will be written at timestep interval =', &
                     tag=diag2_step)
       if (write_step==ci) then
          call Message('Error: Full field snapshot write interval not set - &
               &choose a value for write_step')
          parameters_ok=.false.
       else
          call Message('Snapshots will be written at timestep interval =', &
                     tag=write_step)
       endif

    endif restart

    cnt = cntr

  end subroutine Init_counters

  !************************************************************************

  function Init_filter(filter_type,filter_exp,k_cut,dealiasing,ft,Nexp) &
       result(filter)
    
    !************************************************************************
    ! Set dealiasing mask for isotropic truncation (semicircle is just tangent
    ! to line in '4/3' rule) and combine it with small scale spatial filter.
    ! Parameter 'filtdec' below is set so that filter decays to 
    ! (1+4*pi/nx)**(-1) at max K allowed by de-aliasing form
    ! Initialize small scale filter/de-aliasing mask.  Legal filter types are:
    !
    !   hyperviscous : equivalent to RHS dissipation 
    !                  nu*del^(2*filter_exp)*field, with nu set optimally.
    !                  Requires 'filter_exp'
    !
    !   exp_cutoff   : exponential cutoff filter (see code below)
    !                  Requires 'filter_exp' and 'k_cut'
    !   
    !   none         : none
    !
    ! Character switch 'dealiasing' sets the de-aliasing type:
    !
    !   orszag       : clips corners of spectral fields at 
    !                  |k|+|l| = (4/3)*(kmax+1) as per Orszag '72
    !
    !   isotropic    : clips spectral fields at all points ouside circle
    !                  which inscribes limits of orszag filter (circle
    !                  tangent to |k|+|l| = (4/3)*(kmax+1) ==>
    !                  Kda = sqrt(8/9)*(kmax+1) ).  In this case, the 
    !                  *effective* Kmax = Kda.
    !
    !   none         : none
    !
    ! ft is tuning factor for max filter value, and Nexp sets
    ! exponent on resolution in (1+4*pi/nx**Nexp) (since SUQG requires 
    ! Nexp = 2).
    !
    !************************************************************************
 
    use io_tools,  only: Message, Read_field
    use qg_params, only: kmax,nx,ny,pi,parameters_ok,cr,ci
    use qg_arrays, only: ksqd_,kx_,ky_
    
    character(*),intent(in)            :: filter_type,dealiasing
    real,intent(in)                    :: filter_exp,k_cut,ft,Nexp
    real,dimension(-kmax:kmax,0:kmax)  :: filter
    ! Local
    real                               :: filtdec
    integer                            :: kmax_da

    filter = 1.   ! Most of dynamic field will be unfiltered - multiplied by 1
    filter(-kmax:0,0) = 0.  ! This part of given field given by conjugate sym

    select case (trim(dealiasing))  
    case ('orszag')

       call Message('Using Orszag (non-isotropic) de-aliasing')
       kmax_da = sqrt(8./9.)*(kmax+1)
       where ( abs(kx_)+abs(ky_) >= (4./3.)*(kmax+1) ) filter = 0.

    case ('isotropic')
       
       call Message('Using isotropic de-aliasing')
       kmax_da = sqrt(8./9.)*(kmax+1)
       where ( ksqd_ >= (8./9.)*(kmax+1)**2 ) filter = 0.

    case ('none')

       call Message('Spectral convolutions will not be de-aliased')
       kmax_da = kmax

    case default

       call Message('Error: dealiasing must be one of orszag|isotropic|none; &
                     &yours is'//trim(dealiasing))
       parameters_ok = .false.

    end select

    ! Now set enstrophy filtering **only in region where filter is not zeroed
    ! from de-aliasing above**

    select case (trim(filter_type))  
    case ('hyperviscous') 

       call Message('Using hyperviscous filter')
       if (filter_exp==cr) then
          call Message('Error: filter_exp not set')
          parameters_ok=.false.
       else
          call Message('...with (del**2)**',r_tag=filter_exp)
       endif

       where (filter>0.) 
          filter = 1/(1+ft*(4*pi/nx**Nexp)*(ksqd_/kmax_da**2)**filter_exp)
       endwhere

    case ('exp_cutoff') 

       call Message('Using exponential cutoff filter')
       if (k_cut==cr) then
          call Message('Error: cutoff scale k_cut not set')
          parameters_ok=.false.
       else
          call Message('Cutoff scale k_cut =',r_tag=k_cut)
       endif
       if (filter_exp==cr) then
          call Message('Error: filter_exp not set')
          parameters_ok=.false.
       else
          call Message('Filter exponent filter_exp =',r_tag=filter_exp)
       endif

       filtdec = -log(1+ft*4*pi/nx**Nexp)/(kmax_da-k_cut)**filter_exp
       where ((ksqd_>k_cut**2).and.(filter>0.)) 
          filter = exp(filtdec*(sqrt(ksqd_)-k_cut)**filter_exp)
       end where

    case ('none')

       call Message('No spatial filtering')

    case default ! or filter_type = 'none' -- make that the default in decln.

       call Message('Error: Must select filter_type.  Legal choices are &
                     &filter_type = hyperviscous|exp_cutoff|none')
       parameters_ok=.false.

    end select

  end function Init_filter

  !************************************************************************

  function Init_forcing(forc_coef,forc_corr,kf_min,kf_max,&
                        norm_forcing,force_file,restarting) result(force)

    !************************************************************************
    ! If random Markovian forcing is to be used, check input params and,
    ! if its a restart run, read in forcing from previous timestep
    !************************************************************************
 
    use io_tools,  only: Message, Read_field
    use qg_params, only: kmax,parameters_ok,cr,ci,datadir

    real                                    :: forc_coef,forc_corr
    real,intent(in)                         :: kf_min,kf_max
    character(*),intent(in)                 :: force_file
    logical, intent(in)                     :: norm_forcing,restarting
    complex,dimension(-kmax:kmax,0:kmax)    :: force
    logical                                 :: file_exists

    call Message('Random Markovian forcing on')
    if (forc_coef==cr) then
       call Message('Error: forc_coef must be set for RM forcing')
       parameters_ok=.false.
    else
       call Message('Forcing with coefficient forc_coef =',r_tag=forc_coef)
    endif
    if (forc_corr==cr) then
       call Message('Info: forc_corr not set - setting to .5')
       forc_corr = .5
    endif
    if ((forc_corr<0).or.(forc_corr>1)) then
       call Message('Error: require 0<=forc_corr<=1 - &
            &yours is:',r_tag=forc_corr)
       parameters_ok=.false.
    endif
    if (kf_min==cr)  then
       call Message('Error: kf_min must be set for RM forcing')
       parameters_ok=.false.
    else
       call Message('Minimum forcing wavenumber kf_min =',tag=int(kf_min))
    endif
    if (kf_max==cr)  then
       call Message('Error: kf_max must be set for RM forcing')
       parameters_ok=.false.
    else
       call Message('Maximum forcing wavenumber kf_max =',tag=int(kf_max))
    endif
    if (norm_forcing) call Message('Info: generation rate set to forc_coef')
    
    if (restarting) then
       inquire(file=trim(datadir)//trim(force_file)//'.bin',exist=file_exists)
       if (file_exists) then
          call Read_field(force,force_file)
       else
          force = 0.
       endif
    else
       force = 0.
    endif


  end function Init_forcing

  !************************************************************************

  subroutine Init_strat(strat_type,surface_bc)

    !************************************************************************
    ! If its a multi-level run, read in or create density profile (rho),
    ! layer thicknesses (dz).  Also set inversion matrix psiq,
    ! vertical modes (vmode), defn wavenumbers (kz) and triple interaction
    ! coefficients (tripint matrix)
    !
    ! Legal values of 'strat_type' are:
    !
    !  linear   : uniform stratification
    !
    !  twolayer : two unequal layers - in this case 'deltc' = dz(1)
    !             and vmode, kz and tripint calculated analytically
    !
    !  exp      : exponential stratification (see qg_strat_tools/Make_strat)
    !             with decay scale 'deltc'
    !
    !  stc      : tanh profile for stratification ( " ") with decay scale
    !             'deltc'
    !
    !  read     : read from 'dz_file' and 'rho_file'
    !************************************************************************

    use io_tools,    only: Message, Write_field, Read_field
    use qg_arrays,   only: dz,rho,drho,psiq,kz,vmode,tripint
    use qg_params,   only: dz_file,rho_file,F,Fe,nz,nv,deltc,hf,drt,drb,rho_slope,&
                           read_tripint,psiq_file,vmode_file,kz_file,tripint_file,&
                           dz_in_file,rho_in_file,tripint_in_file,parameters_ok,cr,ci
    use strat_tools, only: Strat_params,Get_vmodes,Get_z,Make_strat,Get_tripint

    character(*),intent(in) :: strat_type, surface_bc
    real,dimension(1:nz)    :: zl

    ! Create or read density (rho) and layer thickness (dz) profiles

    select case (trim(strat_type))
    case ('linear')
          
       call Message('Linear stratification selected')

       deltc = 0.
       dz = 1./float(nz)
       zl = Get_z(dz)
       rho = 1+((zl-zl(1))/(zl(nz)-zl(1))-.5)

    case ('twolayer')
          
       call Message('Two (unequal) layer stratification selected')
       if (nz/=2) then
          call Message('Error: this setting invalid for nz /= 2')
          Parameters_ok=.false.
       endif
       if (deltc==cr) then
          call Message('Error: deltc must be set to give top layer thickness')
          Parameters_ok=.false.
       else
          call Message('Upper layer thickness ratio deltc =',r_tag=deltc)
       endif

       dz(1) = deltc; dz(2) = 1-deltc
       zl = Get_z(dz)
       rho = 1+((zl-zl(1))/(zl(nz)-zl(1))-.5)

    case('exp')
          
       call Message('Exponential stratification selected')
       if (nz==2) &
            call Message('Caution: consider using strat_type = twolayer &
            &which interprets deltc as upper layer thickness fraction')
       if (deltc==cr) then
          call Message('Error: deltc must be set to make EXP rho')
          Parameters_ok=.false.
       else
          call Message('Scale depth deltc =',r_tag=deltc)
       endif
          
       call Make_strat(dz,rho,deltc,strat_type,rho_slope)

    case ('stc')
          
       call Message('Surface intensified stratification selected')
       if (nz==2) &
            call Message('Caution: consider using strat_type = twolayer &
            &which interprets deltc as upper layer thickness fraction')
       if (deltc==cr) then
          call Message('Error: deltc must be set to make STC rho')
          Parameters_ok=.false.
       else
          call Message('Scale depth deltc =',r_tag=deltc)
       endif

       call Make_strat(dz,rho,deltc,strat_type,rho_slope)

    case ('read')
          
       if (trim(rho_file)=='') then
          call Message('Error: no input file for rho given')
          Parameters_ok=.false.
       endif
       if (trim(dz_file)=='') then
          call Message('Error: no input file for dz given')
          Parameters_ok=.false.
       endif
       call Message('Will read density from: '//trim(rho_file))
       call Message('Will read layer thicknesses from: '//trim(dz_file))
       
       call Read_field(dz,dz_in_file,exclude_dd=1)
       call Read_field(rho,rho_in_file,exclude_dd=1)

       dz = dz/sum(dz) 
       rho = rho/sum(rho*dz) 

    case default
          
       call Message('Error: must select strat_type = &
            &linear|stc|exp|read &
            &with nz>1 - yours is:'//trim(strat_type))
       Parameters_ok=.false.
       
    end select

    drho(1:nz-1) = rho(2:nz)-rho(1:nz-1)       ! Get delta_rho
    if (trim(surface_bc)=='surf_buoy') then ! Get drho to surf buoy layer
!       drho(0) = drho(1)*dz(1)/(dz(1)+dz(2))
       drho(0) = drho(1)*dz(1)/(dz(1)+dz(2))
    endif
    drho = drho/(sum(drho)/size(drho))         ! Normalize drho
    call Get_vmodes(dz,drho(1:nz-1),F,Fe,kz,vmode,surface_bc)
    if (read_tripint) then
       call Read_field(tripint,tripint_in_file,exclude_dd=1)
    else
       tripint = Get_tripint(dz,rho,surface_bc,hf,drt,drb)
    endif
    psiq = strat_params(dz,drho,F,Fe,surface_bc,nv)

    ! Save strat fields for post-processing

    call Write_field(psiq,psiq_file)
    call Write_field(vmode,vmode_file)
    call Write_field(kz,kz_file)            
    call Write_field(dz,dz_file)
    call Write_field(rho,rho_file)
    call Write_field(tripint,tripint_file)

  end subroutine Init_strat

  !************************************************************************

  function Init_ubar(ubar_type,ubar_in_file,uscale,umode,ubar_file) result(ubar)

    !************************************************************************
    ! Make mean zonal velocity profile. Legal 'ubar_type' values are
    !
    !  stc     : half gaussian profile with decay scale 'delu'
    !
    !  exp     : exponential profile with decay scale 'delu'
    !  
    !  linear  : linearly varying profile (Eady profile)
    !
    !  modal   : ubar projects exactly onto mode 'umode'
    !
    !  read    : read from file 'ubar_in_file'
    !
    ! Result of any init type is multiplied by uscale.  All init types
    ! except 'read' and 'modal' are normalized by Normu.
    !************************************************************************

    use io_tools,    only: Message, Write_field, Read_field
    use qg_arrays,   only: dz,vmode
    use qg_params,   only: nz,delu,parameters_ok,cr,ci
    use strat_tools, only: Get_z

    character(*),intent(in)  :: ubar_type,ubar_in_file,ubar_file
    integer,intent(in)       :: umode
    real,intent(inout)          :: uscale
    real,dimension(1:nz)     :: ubar
    real,dimension(1:nz)     :: zl

    zl = Get_z(dz)
    select case (trim(ubar_type))
    case ('stc')

       call Message('Surface intensified mean zonal velocity selected')
       if (delu==ci) then
          call Message('Error: delu must be set for making ubar profile')
          Parameters_ok=.false.
       elseif (delu<0) then
          call Message('Error: require delu>=0 - yours is:',r_tag=delu)
          Parameters_ok=.false.
       endif

       ubar = Normu(exp(-(zl/delu)**2),dz)

    case ('exp')

       call Message('Exponential mean zonal velocity selected')
       if (delu==ci) then
          call Message('Error: delu must be set for making ubar profile')
          Parameters_ok=.false.
       elseif (delu<0) then
          call Message('Error: require delu>=0 - yours is:',r_tag=delu)
          Parameters_ok=.false.
       endif

       ubar = Normu(exp(-zl/delu),dz)

    case ('linear')

       call Message('Linear mean zonal velocity selected')
       delu = 0.

       ubar = Normu(2*(1-(zl-zl(1))/(zl(nz)-zl(1)))-1,dz)

    case ('modal') 

       call Message('Modal mean zonal velocity selected')
       if (umode==ci) then
          call Message('Error: umode must be set for modal ubar')
          Parameters_ok=.false.
       elseif ((umode<0).or.(umode>nz-1)) then
          call Message('Error: require 0<umode<=nz; yours is:',tag=umode)
          Parameters_ok=.false.
       endif

       ubar = vmode(:,umode+1)

    case ('read')

       if (trim(ubar_file)=='') then
          call Message('Error: no input file for ubar given')
          Parameters_ok=.false.
       endif
       call Message('Will read mean U from: '//trim(ubar_file))

       call Read_field(ubar,ubar_in_file,exclude_dd=1)

    case('none') 

       call Message('Zero mean shear selected')
       uscale=0.

    case default

       call Message('Error: must select ubar_type = linear|stc|exp|modal|&
            &read with nz>1 - yours is:'//trim(ubar_type))
       Parameters_ok=.false.

    end select
    
    ubar = ubar*uscale
    
    call Write_field(ubar,ubar_file)

  end function Init_ubar

  !*********************************************************************

  function Normu(ubarin,dz) result(ubarout)

    !************************************************************************
    ! Normalize initial mean velocities such that (1) there is no barotropic
    ! component, (2) ubar = ubar/Int(abs(ubar)dz), (3) ubar at top is 
    ! positive.  Overall scale and sign of ubar is then set by 'uscale'
    !************************************************************************
    
    real,dimension(:),intent(in)   :: ubarin, dz
    real,dimension(size(ubarin,1)) :: ubarout

    ubarout = ubarin - sum(ubarin*dz)
    if (sum(abs(ubarout)*dz)/=0) ubarout = ubarout/sum(abs(ubarout)*dz)
    if (ubarout(1)<0) ubarout=-ubarout ! sign of uscale will be sign of top u
    
  end function Normu

  !************************************************************************

  function Init_topo(topo_type,restarting) result(hb)

    !************************************************************************
    ! Read in or create bottom topography in a manner specified by 
    ! 'topo_type', which can have the values:
    !   
    !   gaussbump : Gaussian bump in center of domain with width 'del_topo'.
    !               Note that in physical space, -pi<=x,y<=+pi
    !   
    !   spectral  : Random topography whose representation in spectral
    !               space is isotropic and centered at 'k_o_topo' with 
    !               width 'del_topo'
    !
    !   xslope    : Slope in zonal direction 
    !
    !   yslope    : Slope in meridional direction (equiv to beta in bottom 
    !               layer only).
    !
    !   read      : Read in initial spectral topography from 'hb_in_file'
    !
    ! All parameters read in via input namelist file.  Also, magnitude
    ! of topography is set by 'toposcale', regardless of topo type.
    ! If not set, its default value is 1.
    !************************************************************************
  
    use io_tools,        only: Message, Write_field, Read_field
    use qg_params,       only: kmax,nz,nx,ny,nkx,nky,toposcale,hb_in_file, &
                               del_topo,k_o_topo,pi,i,idum,hb_file, &
                               parameters_ok,cr,ci
    use qg_arrays,       only: ksqd_
    use transform_tools, only: Grid2spec
    use numerics_lib,    only: Ran

    character(*),intent(in)                  :: topo_type
    logical, intent(in)                      :: restarting
    complex,dimension(-kmax:kmax,0:kmax)     :: hb
    real,dimension(:,:),allocatable          :: hbg
    integer                                  :: midx, midy, ix, iy
    real                                     :: radius2
    
    if (restarting) then

       call Read_field(hb,hb_file)

    else 

       call Message('Using bottom topography')
       
       midx = kmax+1; midy = kmax+1
       select case (trim(topo_type))
       case('gaussbump')
          
          call Message('Gaussian bump topography selected')
          if (del_topo==cr) then
             call Message('Error: bump width del_topo must be set to make &
                           &gaussian bump topography')
             Parameters_ok=.false.
          endif
          if (del_topo<=0) then
             call Message('Error: require del_topo>=0 - yours is:',&
                           r_tag=del_topo)
             Parameters_ok=.false.
          endif
          
          allocate(hbg(nx,ny)); hbg = 0.
          do iy = 1,ny
             do ix = 1,nx
                radius2 = ((ix-midx)**2+(iy-midy)**2)*((2*pi)**2/(nx*ny))
                hbg(ix,iy) = exp(-radius2/del_topo**2)
             enddo
          enddo
          hb = Grid2spec(hbg)  
          deallocate(hbg)
          
       case('spectral')           ! Make a spectral peak at k_o_topo
          
          call Message('Spectral topography selected')
          if (del_topo==cr) then
             call Message('Error: spectral width del_topo must be set for&
                  & spectral topography')
             Parameters_ok=.false.
          endif
          if (del_topo<=0) then
             call Message('Error: require del_topo>=0 - yours is:',&
                           r_tag=del_topo)
             Parameters_ok=.false.
          endif
          if (k_o_topo==cr) then
             call Message('Error: spectral centroid k_o_topo must be set to &
                  &make spectral topography')
             Parameters_ok=.false.
          endif
          
          hb(:,:) = exp(-(sqrt(ksqd_)-k_o_topo)**2/del_topo**2) &
               *cexp(i*2*pi*Ran(idum,nkx,nky))
          
       case('xslope')
          
          call Message('Linear zonal slope topography selected')
          
          allocate(hbg(nx,ny)); hbg = 0.
          do iy = 1,ny
             do ix = 1,nx
                hbg(ix,iy) = (ix-1)*(2*pi/nx)
             enddo
          enddo
          hb = Grid2spec(hbg)  
          deallocate(hbg)
          
       case('yslope')
          
          call Message('Linear meridional slope topography selected')
          
          allocate(hbg(nx,ny)); hbg = 0.
          do iy = 1,ny
             do ix = 1,nx
                hbg(ix,iy) = (iy-1)*(2*pi/ny)
             enddo
          enddo
          hb = Grid2spec(hbg)  
          deallocate(hbg)
          
       case('read')
          
          if (trim(hb_file)=='') then
             call Message('Error: no input file for topography given')
             Parameters_ok=.false.
          endif
          call Message('Will read spectral topography from: '//trim(hb_file))
          
          call Read_field(hb,hb_in_file,exclude_dd=1)
          
       case default
          
          call Message('Error: must select topo_type = &
                       &spectral|gaussbump|xslope|yslope|read&
                       & - yours is:'//trim(topo_type))
          Parameters_ok=.false.
          
       end select
       
       if (toposcale==cr) then
          call Message('Warning: toposcale not set - setting to 1')
          toposcale = 1.
       endif
       
       hb = toposcale*hb
       
       call Write_field(hb,hb_file)
       
    endif

  end function Init_topo

  !************************************************************************

  function Init_tracer(tracer_init_type,restarting) result(tracer)

    !************************************************************************
    ! Read in or create initial tracer distribution in manner specified 
    ! by 'tracer_init_type', which can have the values:
    !
    !   spatially_constant : spread tracer uniformly over grid 
    !
    !   spatially_centered : put it all in center of grid initially.
    !
    !   read               : read in inital spectral tracer distribution 
    !                        from file 'tracer_init_file'
    !
    ! In all but 'read' case, total initial variance is set by 'tvar_o'.
    ! Default is tvar_o = 0.
    !************************************************************************

    use op_rules,        only: operator(+), operator(-), operator(*)
    use io_tools,        only: Message, Write_field, Read_field
    use qg_params,       only: tracer_init_file,tvar_o,kmax,nx,ny,&
                               frame,tracer_file,parameters_ok,cr,ci, &
                               datadir
    use qg_arrays,       only: filter_t
    use transform_tools, only: Grid2spec
     
    character(*),intent(in)                 :: tracer_init_type
    logical, intent(in)                     :: restarting
    complex,dimension(-kmax:kmax,0:kmax)    :: tracer
    real,dimension(2*(kmax+1),2*(kmax+1))   :: tracerg
    integer                                 :: midx,midy
    real                                    :: tv
    logical                                 :: file_exists

    if (restarting) then

       inquire(file=trim(datadir)//trim(tracer_file)//'.bin',exist=file_exists)
       if (file_exists) then
          call Read_field(tracer,tracer_file,frame+1)
       else 
          tracer = 0.
       endif

    elseif (.not.restarting) then

       midx = kmax+1; midy = midx
       select case (trim(tracer_init_type))    
       case ('')
          tracer = 0.
       case ('spatially_constant')
          call Message('Will spread initial tracer field uniformly over grid')
          tracerg = 1.
          tracer = Grid2spec(tracerg)       
       case ('spatially_centered')
          call Message('Will put all initial tracer field in center of grid')
          tracerg = 0.
          tracerg(midx,midy) = 1.
          tracer = Grid2spec(tracerg)
       case ('read')
          if (trim(tracer_file)=='') then
             call Message('Error: no input file for tracer given')
             Parameters_ok=.false.
          endif
          call Message('Will read initial tracer field from: '&
               &//trim(tracer_init_file))
          call Read_field(tracer,tracer_init_file,exclude_dd=1) 
       case default
          call Message('Error: legal tracer init types are &
               &tracer_init_type=read|spatially_centered|spatially_constant&
               & - yours is:'//trim(tracer_init_type))
          parameters_ok=.false.
       end select

       tracer = filter_t*tracer
       if (tvar_o>0) then
          tv = sum(tracer*conjg(tracer))  ! Normalize init variance = tvar_o
          tracer = sqrt(tvar_o/tv)*tracer
       elseif (tvar_o<0.and.tvar_o/=cr) then
          call Message('Error: require tvar_o >=0')
          Parameters_ok=.false.
       endif
          
    end if

  end function Init_tracer

  !************************************************************************

  function Init_streamfunction(psi_init_type,restarting) result(psi)

    !************************************************************************
    ! Read in or create initial streamfunction field in manner specified
    !   by variable 'psi_init_type', which can have the values:
    !
    ! spectral_m :  Gaussian spread of initial energy about isotropic horiz.
    !               wavenumber 'k_o' and with width 'delk', and all energy
    !               in vertical mode 'm_o'
    ! spectral_z :  Same as above in horizontal plane, but with all initial
    !               energy in level 'z_o'
    ! elliptical_vortex :  Elliptical gaussian bump in initial grid vorticity
    !               field, aspect ratio 'aspect_vort' and width 'del_vort',
    !               and contained in mode 'm_o'
    ! read :        Read in from 'psi_init_file' at frame 'start_frame'
    ! 
    ! All of the values in quotes can be set in input namelist
    !************************************************************************

    use op_rules,        only: operator(+), operator(-), operator(*)
    use io_tools,        only: Message, Write_field, Read_field
    use qg_params,       only: kmax,nz,z_o,k_o,m_o,e_o,delk,i,pi,idum,nkx,nky,&
                               nv,psi_init_file,start_frame,nx,ny,aspect_vort,&
                               del_vort,parameters_ok,cr,ci,psi_file,frame,&
                               time,time_file,initialize_energy
    use qg_arrays,       only: ksqd_,vmode,kz,dz,filter
    use qg_diagnostics,  only: energy
    use strat_tools,     only: Layer2mode, Mode2layer
    use numerics_lib,    only: Ran
    use transform_tools, only: Grid2spec

    character(*),intent(inout)                 :: psi_init_type
    logical, intent(in)                        :: restarting
    complex,dimension(-kmax:kmax,0:kmax,1-(nv-nz):nz)    :: psi
    real,dimension(:,:),allocatable            :: espec, mu
    real,dimension(:,:,:),allocatable          :: zetag
    complex,dimension(:,:,:),allocatable       :: zeta,psim
    real                                       :: e, radius2
    integer                                    :: ix, iy, midx, midy, n

    if (trim(psi_init_type)=='spectral') psi_init_type = 'spectral_m'

    restart: if (restarting) then

       call Read_field(psi,psi_file,frame+1)
       call Read_field(time,time_file,frame+1)   ! Reset clock
       call Message('Will read streamfunction from:'//trim(psi_file)//&
            &', frame:',tag=start_frame+1)

    else 

       psi = 0.
       select case (trim(psi_init_type))
       case('spectral_m')

          call Message('Initial streamfunction will be spectrally local')
          if (k_o==cr) then
             call Message('Error: k_o must be set to make streamfunction')
             Parameters_ok=.false.
          elseif (k_o<=0) then
             call Message('Error: require k_o > 0 - yours is:', r_tag=k_o)
             Parameters_ok=.false.
          else
             call Message('Initial energy centroid at isotropic wavenumber &
                  &k_o =', r_tag=k_o)
          endif
          if (delk==cr) then
             call Message('Error: delk must be set to make streamfunction')
             Parameters_ok=.false.
          elseif (delk<=0) then
             call Message('Error: need delk>0 - yours is:',r_tag=delk)
             Parameters_ok=.false.
          else
             call Message('Initial energy peak wavenumber width delk =',&
                  r_tag=delk)
          endif
          if (nz>1) then
             if (m_o==ci) then
                call Message('Error: need m_o for modal streamfunction')
                Parameters_ok=.false.
             elseif ((m_o<0).or.(m_o>nz-1)) then
                call Message('Error: require 0<=m_o<= nz-1 - yours is:',&
                              tag=m_o)
                Parameters_ok=.false.
             else
                call Message('Streamfunction will be initd in mode m_o =',&
                     tag=m_o)
             endif
          else
             m_o = 0
          endif

          allocate(espec(-kmax:kmax,0:kmax),mu(-kmax:kmax,0:kmax))
          allocate(psim(-kmax:kmax,0:kmax,nz)); psim = 0.
          espec = 1.
          if (delk/=0) espec = exp(-(sqrt(ksqd_)-k_o)**2/delk**2) 
          mu = sqrt(ksqd_+kz(m_o+1)**2)        ! Total geostrophic wavenumber
          psim(:,:,m_o+1) = sqrt(espec)/mu*cexp(i*2*pi*Ran(idum,nkx,nky))
          psi(:,:,1:nz) = Mode2layer(psim,vmode)
          deallocate(espec,mu,psim)
          
       case('spectral_z')
          
          call Message('Initial streamfunction will be spectral by layer')
          if (k_o==cr) then
             call Message('Error: k_o must be set to make streamfunction')
             Parameters_ok=.false.
          elseif (k_o<=0) then
             call Message('Error: require k_o > 0 - yours is:', r_tag=k_o)
             Parameters_ok=.false.
          else
             call Message('Initial energy centroid at isotropic wavenumber &
                  &k_o =', r_tag=k_o)
          endif
          if (delk==cr) then
             call Message('Error: delk must be set to make streamfunction')
             Parameters_ok=.false.
          elseif (delk<0) then
             call Message('Error: require delk>=0 - yours is:',r_tag=delk)
             Parameters_ok=.false.
          else
             call Message('Initial energy peak wavenumber width delk =',&
                  r_tag=delk)
          endif
          if (nz>1) then
             if (z_o==ci) then
                call Message('Error: z_o must be set for streamfunction')
                Parameters_ok=.false.
             elseif ((z_o<=0).or.(z_o>nz)) then
                call Message('Error: need 0<z_o<=nz - yours is:',tag=z_o)
                Parameters_ok=.false.
             else
                call Message('Streamfunction will be initd in layer z_o =',&
                           tag=z_o)
             endif
          else
             z_o = 1
          endif
       
          allocate(espec(-kmax:kmax,0:kmax),psim(-kmax:kmax,0:kmax,nz))
          espec = 1.
          if (delk/=0) espec = exp(-(sqrt(ksqd_)-k_o)**2/delk) 
          psi(:,:,z_o) = sqrt(espec)/sqrt(ksqd_)*cexp(i*2*pi*Ran(idum,nkx,nky))
          deallocate(espec,psim)

       case('elliptical_vortex')

          call Message('Initial vorticity field will be an elliptical vortex')
          if (del_vort==cr) then
             call Message('Error: del_vort must be set to make streamfunction')
             Parameters_ok=.false.
          else
             call Message('Initial vortex width del_vort =',r_tag=del_vort)
          endif
          if (aspect_vort==cr) then
             call Message('Error: aspect_vort must be set to make streamfuncn')
             Parameters_ok=.false.
          else
             call Message('Initial vortex aspect ratio aspect_vort =', &
                  r_tag=aspect_vort)
          endif
          if (nz>1) then
             if (m_o==ci) then
                call Message('Error: m_o must be set for modal streamfunction')
                Parameters_ok=.false.
             elseif ((m_o<0).or.(m_o>nz-1)) then
                call Message('Error: require 0<= m_o<=nz-1: yours is:',tag=m_o)
                Parameters_ok=.false.
             else
                call Message('Streamfunction will be initd in mode m_o =',&
                     tag=m_o+1)
             endif
          else
             m_o = 0
          endif
       
          allocate(zetag(nx,ny,1),zeta(-kmax:kmax,0:kmax,1))
          allocate(psim(-kmax:kmax,0:kmax,nz))
          midx = kmax+1; midy = midx; zetag = 0.; psim = 0.; zeta = 0.
          do iy = 1,ny
             do ix = 1,nx
                radius2 = ((ix-midx)**2+aspect_vort*(iy-midy)**2)
                radius2 = radius2*((2*pi)**2/(nx*ny))
                zetag(ix,iy,1) = exp(-radius2/del_vort**2)
             enddo
          enddo
          zeta = Grid2spec(zetag)
          psim(:,:,m_o+1) = -zeta(:,:,1)*(1./ksqd_)
          psi(:,:,1:nz) = Mode2layer(psim,vmode)
          deallocate(zetag,zeta,psim)
          
       case('read')
          
          if (trim(psi_init_file)=='') then
             call Message('Error: no input file for psi given')
             Parameters_ok=.false.
          endif
          if (start_frame==ci) then
             call Message('Warning: start_frame not initialized-setting to 1')
             start_frame = 1
          elseif (start_frame<=0) then
             call Message('Error: require start_frame>=0')
             Parameters_ok=.false.
          endif
          call Message('Initial streamfunction will be read from: '&
               &//trim(psi_init_file)//', frame:', tag=start_frame)
          
          if (initialize_energy) then
             call Message('Your input field will be normalized - set&
                  & initialize_energy = F to prevent this')
          endif

          call Read_field(psi,psi_init_file,frame=start_frame,exclude_dd=1)
          
       case default
          
          call Message('Error: must select psi_init_type = &
               &read|spectral_m|spectral_z|spectral|elliptical_vortex &
               &- yours is:'//trim(psi_init_type))
          Parameters_ok=.false.
          
       end select

       psi = filter*psi
       if (initialize_energy) then ! Init energy to e_o
          if (e_o==cr) then
             call Message('Info: e_o not set - setting to 1.')
          elseif (e_o<=0) then
             call Message('Error: must have e_o > 0')
             parameters_ok = .false.
          else
             call Message('Initial energy will be set to e_o =', r_tag=e_o)
          endif
          if (energy(psi)>epsilon(e_o)) then
             psi = sqrt(e_o/energy(psi))*psi 
          else
             call Message('Error: no amplitude in initial psi field:')
          endif
       endif

    endif restart

  end function Init_streamfunction

  !*********************************************************************

end module qg_init_tools
