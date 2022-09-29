module module_random_field
! log 15-2-2011
! The wind perturbations are spatiotemporally correlated in both options but they are “non-divergent” in rf_prsflag = 2.
! temporal correlate all 4 atmospheric variables.
! lognormal format is applied for snowfall rate
!
! ----------------------------------------------------------------------------
! -- init_rand_update - Initializes the random module, reads infile2.in - sets
!                       array sizes of derived types and initializes variables
!                       ran1 and ran. Also sets random number seed, and sets
!                       the FFT sizes in mod_pseudo
! -- rand_update      - main routine - updates the nondimensional perturbations
!                       contained in variable "ran", and applies the
!                       dimensionalized version of this to the different forcing
!                       fields contained in "forcing_fields". Note that this
!                       routine is (for now) applied only when hf forcing is
!                       enabled (a check is done against forcing update times
!                       "rdtime" given by mod_forcing_nersc and yrflag). Also,
!                       synoptic flags from "mod_forcing_nersc" are not taken
!                       into account.
!
! + Various private routines:
! -- limits_randf     - reads infile2.in and sets forcing variances as well as
!                       spatial and temporal correlation lengths (private)
! -- set_random_seed2 - Sets the random number seed based on date and time
! -- ranfields          Produces a set of nondimensional perturbation fields in
!                       a forcing_fields type
! -- calc_forc_update   Creates a dimensional forcing_field type
! -- assign_force       sets  forcing_field to a constant
! -- assign_vars        Sets variances to a constant
! -- ran_update_ran1    Updates nondimensional forcing_field "ran" using
!                       input variances and nondimensional stochastic forcing
!                       "ran1" - produces a red timeseries specified by input
!                       "alpha"
! -- init_ran(ran)      Allocates variables in the forcing_fields type
! ----------------------------------------------------------------------------

   implicit none
   private

   logical, save :: debug=.false. ! Switches on/off verbose
   real   , save :: rf_hradius  ! Horizontal decorr length for rand forc [km]
   real   , save :: rf_tradius  ! Temporal decorr length for rand forc
   real   , save :: rh          ! Horizontal decorr length for rand forc [grid cells]
   real   , save :: rv
   integer, save :: rf_prsflg=2 ! random forcing pressure flag
   integer, save :: idm, jdm    ! domain size, from inputs
   real,parameter     :: airdns  =  1.2
   real, parameter    :: radian  = 57.2957795
   real, parameter    :: pi      =  3.1415926536
   real, parameter    :: radtodeg= 57.2957795
   real   , save :: dx !  dx is the same as minscpx. this code is based on HYCOM, which the grid size variables are scpx / scux / scvy
   real, allocatable, dimension(:,:)  :: synuwind, synvwind,synsnowfall,syndwlongw,synsss,synsst !,synwndspd
                                       !  ,synairtmp, synrelhum, synprecip,       &
                                       !   synclouds, syntaux, syntauy, synslp,   &
                                       !   synradflx, synshwflx, synvapmix, synssr   ! commented variables.

  ! Random forcing variables:
   type forcing_fields
      real, pointer ::  slp(:,:) !  Sea level pressure
      real, pointer ::  wndspd(:,:) !  wind speed (tke source), it affects x-y wind speed components by wprsfac=sqrt(vars%wndspd)/(3*wprsfac)
      real, pointer ::  snowfall(:,:) !  snow fall rate
      real, pointer ::  dwlongw(:,:) !  longwave downwelling radiation rate
      real, pointer ::  sss    (:,:) !  SSS for relax
      real, pointer ::  sst    (:,:) !  SST for relax
      ! real,pointer ::  uwind  (:,:) !  u-component of wind
      ! real,pointer ::  vwind  (:,:) !  v-component of wind
      ! real,pointer ::  taux   (:,:) !  wind stress in x direction
      ! real,pointer ::  tauy   (:,:) !  wind stress in y direction
      ! real,pointer ::  airtmp (:,:) !  pseudo air temperature
      ! real,pointer ::  relhum (:,:) !  relative humidity
      ! real,pointer ::  clouds (:,:) !  cloud cover
      ! real,pointer ::  precip (:,:) !  precipitation
      ! real,pointer ::  tauxice(:,:) !  ice stress on water in x dir
      ! real,pointer ::  tauyice(:,:) !  ice stress on water in y dir
   end type forcing_fields

   type forcing_variances
      real slp
      real wndspd
      real snowfall
      real dwlongw
      real sss
      real sst
      ! real taux
      ! real tauy
      ! real airtmp
      ! real relhum
      ! real clouds
      ! real precip

   end type forcing_variances

   ! Variable containing forcing variances
   type(forcing_variances), save :: vars

   ! Variable containing nondimensional random forcing fields
   type(forcing_fields), save :: ran, ran1

   interface assignment(=)
      module procedure assign_force
      module procedure assign_vars
   end interface

   interface sqrt
      module procedure var_sqrt
   end interface

   public :: init_rand_update, rand_update, init_fvars, limits_randf

contains

   subroutine init_rand_update(synforc) !, previous_perturbation_exist)
      use module_pseudo2d
      implicit none
      !integer:: previous_perturbation_exist
      real*8, dimension(idm,jdm,6) :: synforc ! dimensional and nondimensional forcing_fields

      if (debug) write (*, '("pseudo-random forcing is active for ensemble generation")')
      if (debug) print *, 'typical model grid scale =', dx  !dx is the same as minscpx.
      rh = rf_hradius/dx     ! Decorrelation length is rh grid cells, So rh here is in number of grid cells.
      rv = rf_tradius        ! Temporal decorrelation scale (days)

      if (debug) print *, "initialized init_ran"
      call init_ran(ran)
      call init_ran(ran1)
      ran  = 0.
      ran1 = 0.
      ! Init
      call set_random_seed2()
      ! Init fft dimensions in mod_pseudo
      call initfftdim(idm, jdm)

      if (debug) print *, 'generating initial random field...'
      call ranfields(ran, rh)
      call rand_update() ! update ran1, ran (final nondimensional fields, ran1 is temporary variable)
      call save_synforc(synforc)
   end subroutine

!----------------------------------
   subroutine set_random_seed2
      ! Sets a random seed based on the wall clock time
      implicit none

      integer, dimension(8)::val
      integer cnt
      integer sze
      integer, allocatable, dimension(:):: pt  ! keeps random seed

      call DATE_AND_TIME(values=val)
      call RANDOM_SEED(size=sze)
      allocate (pt(sze))
      call RANDOM_SEED         ! Init - assumes seed is set in some way based on clock, date etc. (not specified in f ortran standard)
      call RANDOM_SEED(GET=pt) ! Retrieve initialized seed
      pt = pt*(val(8) - 500)  ! val(8) is milliseconds - this randomizes stuff if random_seed is nut updated often e nough
      call RANDOM_SEED(put=pt)
      deallocate (pt)
   end subroutine set_random_seed2

    !subroutine set_random_seed
    !! Sets a random seed based on the system and wall clock time
    !! An MPI version exists, that spreads the seed to all tiles.
    !    implicit none
    !    integer, dimension(8)::val
    !    integer cnt
    !    integer sze
    !    integer, allocatable, dimension(:):: pt

    !    call DATE_AND_TIME(values=val)
    !    call SYSTEM_CLOCK(count=cnt)
    !    call RANDOM_SEED(size=sze)
    !    allocate (pt(sze))
    !    pt(1) = val(8)*val(3)
    !    pt(2) = cnt
    !    call RANDOM_SEED(put=pt)
    !    deallocate (pt)
    !end subroutine set_random_seed

!c --- Initialize FFT dimensions used in pseudo routines
   subroutine initfftdim(nx, ny)
      use module_pseudo2d
      implicit none
      integer, intent(in) :: nx, ny
      fnx = ceiling(log(float(nx))/log(2.))
      fnx = 2**fnx
      fny = ceiling(log(float(ny))/log(2.))
      fny = 2**fny
      if (debug) write (*, '("Fourier transform dimensions ",2i6)') fnx, fny  ! fourier transform dimensions
   end subroutine

   subroutine limits_randf(xdim, ydim)

      ! &pseudo2d          !namelist
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
      implicit none
      integer :: seed, prsflg, xdim, ydim
      real    :: vslp, vwndspd !vtaux, vtauy, vclouds
      real    :: scorr, scorr_dx, tcorr, vsnowfall, vdwlongw, vsss, vsst !, vairtmp, vprecip, vrelhum
      character(80) :: nmlfile, cwd

      namelist /pseudo2d/ vslp, vwndspd, vsnowfall, vdwlongw, vsss, vsst, &
         scorr, scorr_dx, tcorr, prsflg

      idm = xdim
      jdm = ydim

      nmlfile = 'pseudo2d.nml'    ! name of general configuration namelist file
      open (99, file=nmlfile, status='old', action='read')
      read (99, NML=pseudo2d)
      close (99)

      ! variables below are defined as global variables in this module: mod_random_forcing
      vars%slp     = vslp
      vars%wndspd  = vwndspd
      vars%snowfall= vsnowfall 
      vars%dwlongw = vdwlongw
      vars%sss     = vsss
      vars%sst     = vsst  
      rf_hradius   = scorr  ! Horizontal decorr length for rand forc [m];
      dx           = scorr_dx
      rf_tradius   = tcorr   ! Temporal decorr length for rand forc; ! tcorr  [grid cells]
      rf_prsflg    = prsflg  ! Pressure flag for random forcing

   end subroutine limits_randf

! --- This routine updates the random forcing component, according to
! --- a simple correlation progression with target variance specified
! --- By forcing_variances. At the end of the routine, if applicable,
! --- the random forcing is added to the forcing fields.
   subroutine rand_update()
      implicit none

      ! rt       -- Information on time (mod_year_info)
      ! ran      -- Nondimensional random perturbations
      ! vars     -- Variances of fields ( Real pert = ran * vars)
      ! lrestart -- Special actions are taken if this is a restart
      !type(forcing_fields)    , intent(inout) :: ran
      !type(forcing_variances) , intent(in)    :: vars

      integer :: ix, jy
      real :: alpha, autocorr, nsteps, wspd, mtime
      !logical, save :: first = .true.
      !logical, save :: lfirst = .true.
      real, parameter :: rhoa = 1.2, cdfac = 0.0012

      real :: cd_new, w4, wfact, wndfac, fcor
      real, dimension(1:idm, 1:jdm) :: dpresx, dpresy
      real :: ucor, vcor, ueq, veq, wcor
      real :: wprsfac, minscpx

      real, parameter :: wlat = 60.
      integer i, j
      real*8, save :: rdtime = 8.d0/24.d0    ! Time step of forcing update

      ! Autocorrelation between two times "tcorr"
      !KAL - quite high? - autocorr = 0.95
      !LB - Still too high      autocorr = 0.75
      autocorr = exp(-1.0)

      ! Number of "forcing" steps rdtime between autocorrelation
      ! decay. Autocorrelation time is tcorr
      nsteps = rv/rdtime

      ! This alpha will guarantee autocorrelation "autocorr"
      ! over the time "rv"
      ! rv -> infinity , nsteps -> infinity, alpha -> 1
      ! rv -> 0        , nsteps -> 0       , alpha -> 0 (when 1>autocorr>0)
      alpha = autocorr**(1/nsteps)   ! alpha = exp(-hf_rdtime/rv)

      !write(lp,*) 'Rand_update -- Random forcing field update'
      ! Add new random forcing field to the newly read
      ! fields from ecmwf or ncep (:,:,4)
      !ran1=sqrt(vars)*ran
      call calc_forc_update(ran1, ran, sqrt(vars))

      if (rf_prsflg .eq. 1 .or. rf_prsflg .eq. 2) then
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         ! rf_prsflag=0 : wind and slp are uncorrelated
         ! rf_prsflag=1 : wind perturbations calculated from slp, using coriolis
         !                parameter at 40 deg N
         ! rf_prsflag=2 : wind perturbations calculated from slp, using coriolis
         !                parameter at 40 deg N, but limited by the setting of
         !                windspeed, to account for the horizontal scale of pert.
         !                As far as the wind is concerned, this is the same as
         !                reducing pressure perturbations
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

         !      minscpx=minval(scpx)
         minscpx = dx ! scalar minscpx is uniform in x-y directions
         wprsfac = 1.
         ! flag used in prsflg=2
         if (rf_prsflg == 2) then
            fcor = 2*sin(40./radtodeg)*2*pi/86400; ! Constant

            ! typical pressure gradient
            wprsfac = 100.*sqrt(vars%slp)/(rh*minscpx)

            ! results in this typical wind magnitude
            wprsfac = wprsfac/fcor

            ! but should give wind according to vars%wndspd
            ! this is a correction factor for that
            wprsfac = sqrt(vars%wndspd)/(3*wprsfac)   ! 3*wprsfac is tuned here, compared with wprsfac used in https://svn.nersc.no/hycom/browser/HYCOM_2.2.37/CodeOnly/src_2.2.37/nersc/mod_random_forcing.F
         end if

         ! Pressure gradient. Coversion from mBar to Pa
         dpresx = 0.
         dpresy = 0.
         do jy = 2, jdm
         do ix = 2, idm
            dpresx(ix, jy) = 100.*(ran1%slp(ix, jy) - ran1%slp(ix - 1, jy))/minscpx*wprsfac
            dpresy(ix, jy) = 100.*(ran1%slp(ix, jy) - ran1%slp(ix, jy - 1))/minscpx*wprsfac  ! scalar minscpx is uniform in x-y directions
         end do
         end do

         do jy = 1, jdm
         do ix = 1, idm
            !fcor=2*sin(max(abs(plat(ix,jy)),20.)/radtodeg)*2*pi/86400;
            fcor = 2*sin(40./radtodeg)*2*pi/86400; ! Coriolis balance (at 40 deg)
            fcor = fcor*rhoa
            vcor = dpresx(ix, jy)/(fcor)
            ucor = -dpresy(ix, jy)/(fcor)

            ! In the equatorial band u,v are aligned with the
            ! pressure gradients. Here we use the coriolis
            ! factor above to set it up (to limit the speeds)
            ueq = -dpresx(ix, jy)/abs(fcor)
            veq = -dpresy(ix, jy)/abs(fcor)

            ! Weighting between coriiolis/equator solution
            !         wcor=sin(min(abs(plat(ix,jy)),wlat) / wlat * pi * 0.5)
            wcor = sin(wlat)/wlat*pi*0.5
            !
            synuwind(ix, jy) = wcor*ucor + (1.-wcor)*ueq
            synvwind(ix, jy) = wcor*vcor + (1.-wcor)*veq
            !synwndspd(ix,jy) = sqrt(synuwind(ix,jy)**2 + synvwind(ix,jy)**2)
            ! The rest use fields independent of slp
            ! lognormal format, note in the original code for TOPAZ, exp term is multiplied to the corresponding field. The * operator is applied in addPerturbation(), but for other variables, the current perturbation is the summarized with previous perturbation.
            synsnowfall(ix, jy) = exp(ran1%snowfall(ix, jy) - 0.5*vars%snowfall**2)
            syndwlongw(ix, jy) = ran1%dwlongw(ix, jy)
            syndwlongw(ix, jy) = max(syndwlongw(ix, jy), 0.0)
            synsss(ix,jy) = ran1%sss(ix,jy)    
            synsst(ix,jy) = ran1%sst(ix,jy)
            ! synslp(ix,jy) = ran1%slp(ix,jy)
            ! synairtmp(ix,jy) = ran1%airtmp(ix,jy)
            ! synrelhum(ix,jy) = ran1%relhum(ix,jy)
            ! synprecip(ix,jy) = ran1%precip(ix,jy)
            ! synclouds(ix,jy) = ran1%clouds(ix,jy)
            ! synprecip(ix,jy) = synprecip(ix,jy) & ! lognormal precip
            ! *exp(ran1%precip(ix,jy) - 0.5*vars%precip**2)
            ! ran1 are the time-correlated random fields, !the -0.5*var^2 term is a bias correction.
            ! synprecip(ix,jy) = max(synprecip(ix,jy),0.0)
            ! synrelhum(ix,jy) = min(max(synrelhum(ix,jy),0.0),1.0)
            ! synclouds(ix,jy) = min(max(synclouds(ix,jy),0.0),1.0) ! restricted between 0 and 100%
         end do
         end do
         ! todo: adding perturbation to wind field systemically increases wind speed.
         ! this increment of wind speed should be reduced with air drag coef. to aviod over estimate the wind forcing. air drag = air drag/mean(windspd)* mean(windspd+perturbations)
         !      synuwind = synuwind - compute_mean(synuwind)
         !      synvwind = synvwind - compute_mean(synvwind)

         ! ! Drag,  New drag - Computed directly from winds now
         ! do jy=2,jdm-1
         ! do ix=2,idm-1
         !    wndfac=(1.+sign(1.,synwndspd(ix,jy)-11.))*.5
         !    cd_new=(0.49+0.065*synwndspd(ix,jy))*1.0e-3*wndfac+cdfac*(1.-wndfac)

         !    w4    =.25*( &
         !       synvwind(ix-1,jy+1)+synvwind(ix,jy+1)+  &
         !       synvwind(ix-1,jy  )+synvwind(ix,jy  ))
         !    wfact=sqrt( synuwind(ix,jy)*synuwind(ix,jy)+w4*w4)* airdns*cd_new
         !    syntaux(ix,jy)=synuwind(ix,jy)*wfact

         !    w4   =.25*( &
         !       synuwind(ix  ,jy-1)+synuwind(ix+1,jy-1)+ &
         !       synuwind(ix+1,jy  )+synuwind(ix  ,jy  ))
         !    wfact=sqrt( synvwind(ix,jy)*synvwind(ix,jy)+w4*w4)* airdns*cd_new
         !    syntauy(ix,jy)=synvwind(ix,jy)*wfact
         ! end do
         ! end do

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! rf_prsflag=0 : wind and slp are uncorrelated
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         ! else  ! rf_prsflg .eq. 0
         ! do jy=2,jdm-1
         ! do ix=2,idm-1
         ! !if (ip(ix,jy)==1) then
         !    syntaux  (ix,jy) = syntaux  (ix,jy) + ran1%taux(ix,jy)
         !    syntauy  (ix,jy) = syntauy  (ix,jy) + ran1%tauy(ix,jy)
         !    ! KAL -- winds are nonlinear functions of tau and mainly
         !    ! KAL -- used for sea ice
         !    wspd = sqrt(syntaux(ix,jy)**2 + syntauy(ix,jy)**2)
         !    wspd = max(sqrt(wspd / (cdfac*rhoa)),0.1)
         !    synuwind(ix,jy) = syntaux (ix,jy) / (wspd*cdfac*rhoa)
         !    synvwind(ix,jy) = syntauy (ix,jy) / (wspd*cdfac*rhoa)
         !    synairtmp(ix,jy) = synairtmp(ix,jy)+ran1%airtmp(ix,jy)
         !    synwndspd(ix,jy) = synwndspd(ix,jy)+ran1%wndspd(ix,jy)
         !    synrelhum(ix,jy) = synrelhum(ix,jy)+ran1%relhum(ix,jy)
         !    synclouds(ix,jy) = synclouds(ix,jy)+ran1%clouds(ix,jy)
         !    synprecip(ix,jy) = synprecip(ix,jy) & ! Lognormal precipitation
         !            *exp(ran1%precip(ix,jy) - 0.5*vars%precip**2)
         !    synrelhum(ix,jy) = min(max(synrelhum(ix,jy),0.0),1.0)
         !    synprecip(ix,jy) = max(synprecip(ix,jy),0.0)
         !    synwndspd(ix,jy) = max(synwndspd(ix,jy),0.0)
         !    synclouds(ix,jy) = min(max(synclouds(ix,jy),0.0),1.0)
         ! !end if
         ! end do
         ! end do
      end if ! rf_prsflg

      ! create a new "Brownian increment" saves to ran1
      call ranfields(ran1, rh)
      ! update ran = alpha*ran + sqrt(1-alpha*alpha)* ran1
      call ran_update_ran1(ran, ran1, alpha)
   end subroutine rand_update
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   subroutine ranfields(ranfld, scorr)
      use module_pseudo2d
      implicit none

      type(forcing_fields), intent(inout) :: ranfld
      real, intent(in)    :: scorr
      ! real, dimension(idm,jdm) :: gtmp, tmp

      ranfld = 0.
      call pseudo2d(ranfld%slp, idm, jdm, 1, scorr, fnx, fny)
      call pseudo2d(ranfld%wndspd, idm, jdm, 1, scorr, fnx, fny)
      call pseudo2d(ranfld%snowfall, idm, jdm, 1, scorr, fnx, fny)
      call pseudo2d(ranfld%dwlongw, idm, jdm, 1, scorr, fnx, fny)
      call pseudo2d(ranfld%sss,idm,jdm,1,scorr,fnx,fny)
      call pseudo2d(ranfld%sst,idm,jdm,1,scorr,fnx,fny)
      ! call pseudo2d(ranfld%taux,idm,jdm,1,scorr,fnx,fny)
      ! call pseudo2d(ranfld%tauy,idm,jdm,1,scorr,fnx,fny)
      ! call pseudo2d(ranfld%airtmp,idm,jdm,1,scorr,fnx,fny)
      ! call pseudo2d(ranfld%relhum,idm,jdm,1,scorr,fnx,fny)
      ! call pseudo2d(ranfld%clouds,idm,jdm,1,scorr,fnx,fny)
      ! call pseudo2d(ranfld%precip,idm,jdm,1,scorr,fnx,fny)
   end subroutine ranfields

!------------------------------------------
   subroutine calc_forc_update(A, B, C)
      type(forcing_fields), intent(inout) :: A
      type(forcing_fields), intent(in)    :: B
      type(forcing_variances), intent(in) :: C
      integer :: i, j
      do j = 1, jdm
      do i = 1, idm
         A%slp   (i,j)  = C%slp    * B%slp   (i,j)
         A%wndspd(i,j)  = C%wndspd * B%wndspd(i,j)
         A%snowfall(i,j)= C%snowfall * B%snowfall(i,j)
         A%dwlongw(i,j) = C%dwlongw* B%dwlongw(i,j)
         A%sss   (i,j)  = C%sss    * B%sss   (i,j)
         A%sst   (i,j)  = C%sst    * B%sst   (i,j)    
         ! A%taux  (i,j)=C%taux   * B%taux  (i,j)
         ! A%tauy  (i,j)=C%tauy   * B%tauy  (i,j)
         ! A%airtmp(i,j)=C%airtmp * B%airtmp(i,j)
         ! A%relhum(i,j)=C%relhum * B%relhum(i,j)
         ! A%clouds(i,j)=C%clouds * B%clouds(i,j)
         ! A%precip(i,j)=C%precip * B%precip(i,j)
      end do
      end do
   end subroutine calc_forc_update

!--------------------------------------------------
   subroutine ran_update_ran1(ran, ran1, alpha)
      type(forcing_fields), intent(inout) :: ran
      type(forcing_fields), intent(in) :: ran1
      real, intent(in) :: alpha

      integer :: ix, jy
      do jy = 1, jdm
      do ix = 1, idm
         ran%slp   (ix,jy)  =alpha*ran%slp   (ix,jy) + sqrt(1-alpha*alpha)*ran1%slp   (ix,jy)
         ran%wndspd(ix,jy)  =alpha*ran%wndspd(ix,jy) + sqrt(1-alpha*alpha)*ran1%wndspd(ix,jy)
         ran%snowfall(ix,jy)=alpha*ran%snowfall(ix,jy) + sqrt(1-alpha*alpha)*ran1%snowfall(ix,jy)
         ran%dwlongw(ix,jy) =alpha*ran%dwlongw(ix,jy) + sqrt(1-alpha*alpha)*ran1%dwlongw(ix,jy)
         ran%sss   (ix,jy)  =alpha*ran%sss   (ix,jy) + sqrt(1-alpha*alpha)*ran1%sss   (ix,jy)
         ran%sst   (ix,jy)  =alpha*ran%sst   (ix,jy) + sqrt(1-alpha*alpha)*ran1%sst   (ix,jy)         
         ! ran%taux  (ix,jy)=alpha*ran%taux  (ix,jy) + sqrt(1-alpha*alpha)*ran1%taux  (ix,jy)
         ! ran%tauy  (ix,jy)=alpha*ran%tauy  (ix,jy) + sqrt(1-alpha*alpha)*ran1%tauy  (ix,jy)
         ! ran%airtmp(ix,jy)=alpha*ran%airtmp(ix,jy) + sqrt(1-alpha*alpha)*ran1%airtmp(ix,jy)
         ! ran%relhum(ix,jy)=alpha*ran%relhum(ix,jy) + sqrt(1-alpha*alpha)*ran1%relhum(ix,jy)
         ! ran%clouds(ix,jy)=alpha*ran%clouds(ix,jy) + sqrt(1-alpha*alpha)*ran1%clouds(ix,jy)
         ! ran%precip(ix,jy)=alpha*ran%precip(ix,jy) + sqrt(1-alpha*alpha)*ran1%precip(ix,jy)
      end do
      end do
   end subroutine

!------------------------------------------
   subroutine init_ran(ran)
   implicit none

      type(forcing_fields), intent(inout) :: ran

      allocate(ran%slp    (idm,jdm))
      allocate(ran%wndspd (idm,jdm))
      allocate(ran%snowfall (idm,jdm))
      allocate(ran%dwlongw (idm,jdm))
      allocate(ran%sss    (idm,jdm))
      allocate(ran%sst    (idm,jdm))
      ! allocate(ran%taux   (idm,jdm))
      ! allocate(ran%tauy   (idm,jdm))
      ! allocate(ran%airtmp (idm,jdm))
      ! allocate(ran%relhum (idm,jdm))
      ! allocate(ran%clouds (idm,jdm))
      ! allocate(ran%precip (idm,jdm))
      ! allocate(ran%tauxice(idm,jdm))
      ! allocate(ran%tauyice(idm,jdm))
   end subroutine

!------------------------------------------
   subroutine init_fvars()
   implicit none

      IF( .NOT. ALLOCATED( synuwind  ) ) allocate(synuwind (idm,jdm))
      IF( .NOT. ALLOCATED( synvwind  ) ) allocate(synvwind (idm,jdm))
      IF( .NOT. ALLOCATED( synsnowfall ) ) allocate(synsnowfall(idm,jdm))
      IF( .NOT. ALLOCATED( syndwlongw ) ) allocate(syndwlongw(idm,jdm))
      IF( .NOT. ALLOCATED( synsss ) ) allocate(synsss(idm,jdm))
      IF( .NOT. ALLOCATED( synsst ) ) allocate(synsst(idm,jdm))
      synuwind(:, :) = 0.
      synvwind(:, :) = 0.
      synsnowfall(:, :) = 1. ! set initial value=1 if using the lognormal format in update_rand
      syndwlongw(:, :) = 0.
      synsss(:,:) = 0.
      synsst(:,:) = 0.

      !   IF( .NOT. ALLOCATED( synwndspd ) ) allocate(synwndspd(idm,jdm))
      !   IF( .NOT. ALLOCATED( synslp    ) ) allocate(synslp   (idm,jdm))
      !   IF( .NOT. ALLOCATED( syntaux   ) ) allocate(syntaux  (idm,jdm))
      !   IF( .NOT. ALLOCATED( syntauy   ) ) allocate(syntauy  (idm,jdm))
      !   IF( .NOT. ALLOCATED( synvapmix ) ) allocate(synvapmix(idm,jdm))
      !   IF( .NOT. ALLOCATED( synairtmp ) ) allocate(synairtmp(idm,jdm))
      !   IF( .NOT. ALLOCATED( synrelhum ) ) allocate(synrelhum(idm,jdm))
      !   IF( .NOT. ALLOCATED( synprecip ) ) allocate(synprecip(idm,jdm))
      !   IF( .NOT. ALLOCATED( synclouds ) ) allocate(synclouds(idm,jdm))
      !   IF( .NOT. ALLOCATED( synradflx ) ) allocate(synradflx(idm,jdm))
      !   IF( .NOT. ALLOCATED( synshwflx ) ) allocate(synshwflx(idm,jdm))
      !   synslp   (:,:)=0.
      !   synwndspd(:,:)=0.
      !   syntaux  (:,:)=0.
      !   syntauy  (:,:)=0.
      !   synvapmix(:,:)=0.
      !   synairtmp(:,:)=0.
      !   synrelhum(:,:)=0.
      !   synprecip(:,:)=0.
      !   synclouds(:,:)=0.
      !   synradflx(:,:)=0.
      !   synshwflx(:,:)=0.
   end subroutine

!------------------------------------------
   subroutine save_synforc(synforc)
      real*8  :: synforc(idm,jdm, 6) ! same dimension size as defined in externaldata.cpp

      synforc(:, :, 1)=synuwind
      synforc(:, :, 2)=synvwind
      synforc(:, :, 3)=synsnowfall
      synforc(:, :, 4)=syndwlongw
      synforc(:, :, 5)=synsss
      synforc(:, :, 6)=synsst

   end subroutine

   ! some of the following functions are not used.
   !------------------------------------------
   real function compute_mean(mat)
      real, dimension(:, :), intent(in) :: mat
      compute_mean = sum(mat)/size(mat)
   end function

   !------------------------------------------
   function var_sqrt(A)
      type(forcing_variances) var_sqrt
      type(forcing_variances), intent(in) :: A
      var_sqrt%slp     = sqrt(A%slp   )
      var_sqrt%wndspd  = sqrt(A%wndspd)
      var_sqrt%snowfall= sqrt(A%snowfall)
      var_sqrt%dwlongw = sqrt(A%dwlongw)
      var_sqrt%sss     = sqrt(A%sss   )
      var_sqrt%sst     = sqrt(A%sst   )
      ! var_sqrt%airtmp= sqrt(A%airtmp)
      ! var_sqrt%relhum= sqrt(A%relhum)
      ! var_sqrt%clouds= sqrt(A%clouds)
      ! var_sqrt%precip= sqrt(A%precip)
      ! var_sqrt%taux  = sqrt(A%taux  )
      ! var_sqrt%tauy  = sqrt(A%tauy  )
   end function var_sqrt

!------------------------------------------
   subroutine assign_force(A, r)
      type(forcing_fields), intent(out) :: A
      real, intent(in) :: r

      integer :: i, j

      do j = 1, jdm
      do i = 1, idm
         A%slp    (i,j) = r
         A%wndspd (i,j) = r
         A%snowfall(i,j) = r
         A%dwlongw (i,j) = r
         A%sss    (i,j) = r
         A%sst    (i,j) = r
         ! A%uwind  (i,j) = r
         ! A%vwind  (i,j) = r
         ! A%taux   (i,j) = r
         ! A%tauy   (i,j) = r
         ! A%airtmp (i,j) = r
         ! A%relhum (i,j) = r
         ! A%clouds (i,j) = r
         ! A%precip (i,j) = r
         ! A%tauxice(i,j) = r
         ! A%tauyice(i,j) = r
      end do
      end do
   end subroutine assign_force

!------------------------------------------
   subroutine assign_vars(A, r)
      type(forcing_variances), intent(out) :: A
      real, intent(in) :: r
      A%slp    = r
      A%wndspd = r
      A%snowfall = r
      A%dwlongw = r
      A%sss    = r
      A%sst    = r
      ! A%taux   = r
      ! A%tauy   = r
      ! A%airtmp = r
      ! A%relhum = r
      ! A%clouds = r
      ! A%precip = r
   end subroutine assign_vars


end module module_random_field
