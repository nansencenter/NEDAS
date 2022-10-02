module module_random_field
! ----------------------------------------------------------------------------
! -- read_nml         - reads pseudo2d.nml and sets forcing variances as well as
!                       spatial and temporal correlation lengths (private)
! -- rand_update      - main routine - updates the nondimensional perturbations
!                       contained in variable "ran", and applies the
!                       dimensionalized version of this to the different forcing
!                       fields contained in "forcing_fields".
! -- set_random_seed  - Sets the random number seed based on date and time
! -- pseudo2d           Produces a set of nondimensional perturbation fields
! ----------------------------------------------------------------------------
implicit none

logical :: debug ! Switches on/off verbose mode
integer :: xdim, ydim, n_sample, n_field
integer :: dx, dt
integer :: prsflg ! random forcing pressure flag
integer :: idm, jdm
real :: rh, rv

type forcing_field
    character(8) :: name  !variable name, max length 8
    real :: vars  ! variances
    real :: hradius ! horizontal decorrelation length scale (km)
    real :: tradius ! temporal decorrelation length (day)
end type forcing_field

type(forcing_field), dimension(100) :: field  !!max n_field = 100 for now

namelist /pseudo2d/ debug, xdim, ydim, dx, dt, n_sample, n_field, field, prsflg

real, allocatable, dimension(:,:) :: ran, ran1  !!nondimensional ran variable


contains

subroutine read_nml()
    implicit none
    integer i

    open (99, file='pseudo2d.nml', status='old', action='read')
    read (99, NML=pseudo2d)
    close (99)

    !!fft likes 2^n grid points, so we use larger grid, then trim it to size
    idm = int(2.**ceiling(log(real(xdim))/log(2.)))
    jdm = int(2.**ceiling(log(real(ydim))/log(2.)))
    if (debug) write(*, '(A,I5,A,I5,A,I5,A,I5,A)') 'xdim=', xdim, ' ydm=', ydim, ' enlarged to idm=', idm, ' jdm=', jdm, ' for generating synforc'

    if (debug) then
        do i=1,n_field
            print *, field(i)%name, field(i)%vars, field(i)%hradius, field(i)%tradius
        enddo
    endif
end subroutine read_nml

subroutine set_random_seed
    implicit none
    integer, dimension(8)::val
    integer cnt
    integer sze
    integer, allocatable, dimension(:):: pt  ! keeps random seed
    ! Sets a random seed based on the wall clock time
    call DATE_AND_TIME(values=val)
    call RANDOM_SEED(size=sze)
    allocate (pt(sze))
    call RANDOM_SEED ! Init - assumes seed is set in some way based on clock
    call RANDOM_SEED(GET=pt) ! Retrieve initialized seed
    pt = pt*(val(8) - 500)  ! val(8) is milliseconds - this randomizes stuff
    call RANDOM_SEED(put=pt)
    deallocate (pt)
end subroutine set_random_seed


! --- This routine updates the random forcing component, according to
! --- a simple correlation progression with target variance specified
! --- By forcing_variances. At the end of the routine, if applicable,
! --- the random forcing is added to the forcing fields.
subroutine rand_update(synforc, i_step)
    use module_pseudo2d
    implicit none
    integer :: ix, jy, i, j, i_step
    real :: alpha, autocorr, wspd, mtime
    real, parameter :: airdns  =  1.2
    real, parameter :: pi      =  3.1415926536
    real, parameter :: radtodeg= 57.2957795
    real, parameter :: rhoa = 1.2, cdfac = 0.0012, wlat=60., plat=40.
    real :: cd_new, w4, wfact, wndfac, fcor
    real, dimension(1:idm, 1:jdm) :: dpresx, dpresy
    real :: ucor, vcor, ueq, veq, wcor
    real :: wprsfac, minscpx
    real*8, dimension(idm,jdm,n_field), intent(inout) :: synforc ! dimensional forcing_fields

    if (debug) write (*, '("pseudo-random forcing is active for ensemble generation")')
    if (debug) print *, 'model grid scale =', dx  !dx is the same as minscpx, km.
    if (debug) print *, 'output time step =', dt  !dt is the output time interval in hours.

    !!random seed from wall clock
    call set_random_seed()

    !! Initialize FFT dimensions used in pseudo routines
    fnx = ceiling(log(float(idm))/log(2.))
    fnx = 2**fnx
    fny = ceiling(log(float(jdm))/log(2.))
    fny = 2**fny
    if (debug) write (*, '("Fourier transform dimensions ",2i6)') fnx, fny

    !!nondimensional random field
    if (debug) print *, 'generating initial random field...'

    do i=1,n_field !!loop over n_field

        allocate(ran(idm,jdm))
        allocate(ran1(idm,jdm))
        ran  = 0.
        ran1 = 0.

        rh = field(i)%hradius/dx  ! Decorrelation length (num of grid cells)
        rv = field(i)%tradius/dt  ! Temporal decorrelation scale (time steps)

        ! Autocorrelation between two times "tcorr"
        !KAL - quite high? - autocorr = 0.95
        !LB - Still too high      autocorr = 0.75
        autocorr = exp(-1.0)

        ! Number of "forcing" steps rdtime between autocorrelation
        ! decay. Autocorrelation time is tcorr

        ! This alpha will guarantee autocorrelation "autocorr"
        ! over the time "rv"
        ! rv -> infinity , alpha -> 1
        ! rv -> 0        , alpha -> 0 (when 1>autocorr>0)
        alpha = autocorr**(1/rv)

        !!previous random field
        if (i_step==0) then
            call pseudo2d(ran1, idm, jdm, 1, rh, fnx, fny)
            ran1 = sqrt(field(i)%vars)*ran1
        else
            ran1 = synforc(:, :, i)
        end if

        !!new random field add current time step
        call pseudo2d(ran, idm, jdm, 1, rh, fnx, fny)
        ran = sqrt(field(i)%vars)*ran

        ran = alpha*ran1 + sqrt(1-alpha**2)*ran

        synforc(:, :, i) = ran

        deallocate(ran, ran1)

    !write(lp,*) 'Rand_update -- Random forcing field update'
    ! Add new random forcing field to the newly read
    ! fields from ecmwf or ncep (:,:,4)
    !ran1=sqrt(vars)*ran
    !call calc_forc_update(ran1, ran, sqrt(vars))

    !if (prsflg .eq. 1 .or. prsflg .eq. 2) then
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! rf_prsflag=0 : wind and slp are uncorrelated
! rf_prsflag=1 : wind perturbations calculated from slp, using coriolis
!                parameter at plat deg N
! rf_prsflag=2 : wind perturbations calculated from slp, using coriolis
!                parameter at plat deg N, but limited by the setting of
!                windspeed, to account for the horizontal scale of pert.
!                As far as the wind is concerned, this is the same as
!                reducing pressure perturbations
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        !      minscpx=minval(scpx)
        !minscpx = dx ! scalar minscpx is uniform in x-y directions
        !wprsfac = 1.
         !flag used in prsflg=2
        !if (prsflg == 2) then
        !fcor = 2*sin(plat/radtodeg)*2*pi/86400; ! Coriolis Constant

        ! typical pressure gradient
        !wprsfac = 100.*sqrt(vars%slp)/(rh*minscpx)

        ! results in this typical wind magnitude
        !wprsfac = wprsfac/fcor

        ! but should give wind according to vars%wndspd
        ! this is a correction factor for that
        !wprsfac = sqrt(vars%wndspd)/(3*wprsfac)   ! 3*wprsfac is tuned here, compared with wprsfac used in https://svn.nersc.no/hycom/browser/HYCOM_2.2.37/CodeOnly/src_2.2.37/nersc/mod_random_forcing.F
        !end if

        ! Pressure gradient. Coversion from mBar to Pa
        !dpresx = 0.
        !dpresy = 0.
        !do jy = 2, jdm
        !do ix = 2, idm
        !dpresx(ix, jy) = 100.*(ran1%slp(ix, jy) - ran1%slp(ix - 1, jy))/minscpx*wprsfac
        !dpresy(ix, jy) = 100.*(ran1%slp(ix, jy) - ran1%slp(ix, jy - 1))/minscpx*wprsfac  ! scalar minscpx is uniform in x-y directions
        !end do
        !end do

        !do jy = 1, jdm
        !do ix = 1, idm
        !fcor = 2*sin(plat/radtodeg)*2*pi/86400; ! Coriolis balance (at plat deg)
        !fcor = fcor*rhoa
        !vcor = dpresx(ix, jy)/(fcor)
        !ucor = -dpresy(ix, jy)/(fcor)

        ! In the equatorial band u,v are aligned with the
        ! pressure gradients. Here we use the coriolis
        ! factor above to set it up (to limit the speeds)
        !ueq = -dpresx(ix, jy)/abs(fcor)
        !veq = -dpresy(ix, jy)/abs(fcor)

        ! Weighting between coriiolis/equator solution
        !         wcor=sin(min(abs(plat(ix,jy)),wlat) / wlat * pi * 0.5)
        !wcor = sin(wlat)/wlat*pi*0.5
        !
        !synuwind(ix, jy) = wcor*ucor + (1.-wcor)*ueq
        !synvwind(ix, jy) = wcor*vcor + (1.-wcor)*veq
        !synwndspd(ix,jy) = sqrt(synuwind(ix,jy)**2 + synvwind(ix,jy)**2)
        ! The rest use fields independent of slp
        ! lognormal format, note in the original code for TOPAZ, exp term is multiplied to the corresponding field. The * operator is applied in addPerturbation(), but for other variables, the current perturbation is the summarized with previous perturbation.
        !synsnowfall(ix, jy) = exp(ran1%snowfall(ix, jy) - 0.5*vars%snowfall**2)
        !syndwlongw(ix, jy) = ran1%dwlongw(ix, jy)
        !syndwlongw(ix, jy) = max(syndwlongw(ix, jy), 0.0)
        !synsss(ix,jy) = ran1%sss(ix,jy)
        !synsst(ix,jy) = ran1%sst(ix,jy)
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
        !end do
        !end do
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
         !else  ! prsflg .eq. 0
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
         !end if
        ! end do
        ! end do
    !end if ! prsflg

    ! create a new "Brownian increment" saves to ran1
    !call pseudo2d(ran1(:, :, i), idm, jdm, 1, rh, fnx, fny)
    !ran = alpha*ran + sqrt(1-alpha**2)*ran1

    !apply the alpha
    !call ran_update_ran1(ran, ran1, alpha)


        !!save the final perturbationto synforc
        !synforc(:, :, i) = ran1(:, :, i)

    end do  !!loop n_field

    !prsflat==1

end subroutine rand_update

end module module_random_field
