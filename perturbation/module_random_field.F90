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
integer :: xdim, ydim, n_sample, n_field, nens
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

namelist /pseudo2d/ debug, xdim, ydim, dx, dt, n_sample, nens, n_field, field, prsflg

real, allocatable, dimension(:,:,:) :: ran, ran1


contains

!!YY: read pseudo2d.nml for setting, variance/hradius/tradius are now set
!!      separately for each varible field(i)
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
! --- by vars, hradius, tradius.
subroutine rand_update(synforc, i_step)
    use module_pseudo2d
    implicit none
    integer :: ix, jy, i, j, m, i_step
    real :: alpha, autocorr, wspd, mtime
    real*8, dimension(idm,jdm,nens,n_field), intent(inout) :: synforc ! dimensional forcing_fields
    integer :: slp_id, uwind_id, vwind_id, taux_id, tauy_id
    real, parameter :: airdns  =  1.2
    real, parameter :: pi      =  3.1415926536
    real, parameter :: radtodeg= 57.2957795
    real, parameter :: rhoa = 1.2, cdfac = 0.0012, wlat=60., plat=40.
    real :: fcor, wndvar, wcor, wprsfac
    real, dimension(idm,jdm,nens) :: slp,uwind,vwind,dpresx,dpresy,ucor,vcor,ueq,veq,cd_new,w4,wfact,wndfac,wndspd
    real, dimension(idm,jdm) :: ens_mean, umean,vmean

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

        allocate(ran(idm,jdm,nens))
        allocate(ran1(idm,jdm,nens))
        ran  = 0.
        ran1 = 0.

        rh = field(i)%hradius/dx  ! Decorrelation length (num of grid cells)
        rv = field(i)%tradius/dt  ! Temporal decorrelation scale (time steps)

        ! Autocorrelation between two times "tcorr"
        !KAL - quite high? - autocorr = 0.95
        !LB - Still too high      autocorr = 0.75
        autocorr = exp(-1.0)

        ! This alpha will guarantee autocorrelation "autocorr"
        ! over the time "rv"
        ! rv -> infinity , alpha -> 1
        ! rv -> 0        , alpha -> 0 (when 1>autocorr>0)
        alpha = autocorr**(1/rv)

        !!previous random field
        if (i_step==0) then
            call pseudo2d(ran1, idm, jdm, nens, rh, fnx, fny)
            ran1 = sqrt(field(i)%vars)*ran1
        else
            ran1 = synforc(:, :, :, i)
        end if

        !!new random field for current time step
        call pseudo2d(ran, idm, jdm, nens, rh, fnx, fny)
        ran = sqrt(field(i)%vars)*ran

        !!apply temporal correlation
        ran = alpha*ran1 + sqrt(1-alpha**2)*ran

        !!make sure zero mean
        ens_mean = sum(ran,3)/real(nens)
        do m=1,nens
            ran(:,:,m) = ran(:,:,m)-ens_mean
        end do

        synforc(:, :, :, i) = ran

        deallocate(ran, ran1)

    end do  !!loop n_field

    !!!special wind perturbation options
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! prsflag=0 : wind and slp are uncorrelated
    ! prsflag=1 : wind perturbations calculated from slp, using coriolis
    !             parameter at plat deg N, but limited by the setting of
    !             windspeed, to account for the horizontal scale of pert.
    !             As far as the wind is concerned, this is the same as
    !             reducing pressure perturbations
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (prsflg .eq. 1) then
        slp_id = 0
        uwind_id = 0
        vwind_id = 0
        taux_id = 0
        tauy_id = 0
        do i=1,n_field
            if (field(i)%name .eq. 'slp     ') slp_id=i
            if (field(i)%name .eq. 'uwind   ') uwind_id=i
            if (field(i)%name .eq. 'vwind   ') vwind_id=i
            if (field(i)%name .eq. 'taux    ') taux_id=i
            if (field(i)%name .eq. 'tauy    ') tauy_id=i
        end do
        if (slp_id .eq. 0 .or. uwind_id .eq. 0 .or. vwind_id .eq. 0) then
            print *, 'when prsflg=1, slp, uwind, and vwind are expected in the field list'
            stop
        end if

        wndvar = field(uwind_id)%vars  !!if field(vwind_id)%vars is different it is discarded
        slp = synforc(:, :, :, slp_id)

        wprsfac = 1.

        fcor = 2*sin(plat/radtodeg)*2*pi/86400; ! Coriolis Constant

        ! typical pressure gradient
        wprsfac = 100.*sqrt(field(slp_id)%vars)/(rh*dx)

        ! results in this typical wind magnitude
        wprsfac = wprsfac/fcor

        ! but should give wind according to wndspd variance
        ! this is a correction factor for that
        wprsfac = sqrt(wndvar)/(3*wprsfac)   ! 3*wprsfac is tuned here, compared with wprsfac used in https://svn.nersc.no/hycom/browser/HYCOM_2.2.37/CodeOnly/src_2.2.37/nersc/mod_random_forcing.F

        ! Pressure gradient. Coversion from mBar to Pa
        dpresx(2:idm,:,:) = 100.*(slp(2:idm,:,:) - slp(1:idm-1,:,:))/dx*wprsfac
        dpresx(1,:,:) = dpresx(2,:,:)
        dpresy(:,2:jdm,:) = 100.*(slp(:,2:jdm,:) - slp(:,1:jdm-1,:))/dx*wprsfac
        dpresy(:,1,:) = dpresy(:,2,:)

        fcor = 2*sin(plat/radtodeg)*2*pi/86400
        fcor = fcor*rhoa
        vcor = dpresx/fcor
        ucor = -dpresy/fcor

        ! In the equatorial band u,v are aligned with the
        ! pressure gradients. Here we use the coriolis
        ! factor above to set it up (to limit the speeds)
        ueq = -dpresx/abs(fcor)
        veq = -dpresy/abs(fcor)

        ! Weighting between coriiolis/equator solution
        wcor = sin(wlat)/wlat*pi*0.5

        uwind = wcor*ucor + (1.-wcor)*ueq
        vwind = wcor*vcor + (1.-wcor)*veq
        wndspd = sqrt(uwind**2 + vwind**2)

        ! adding perturbation to wind field systemically increases wind speed.
        ! this increment of wind speed should be reduced with air drag coef.
        ! to aviod over estimate the wind forcing. air drag = air drag/mean(windspd)* mean(windspd+perturbations)
        !!YY: make sure wind pert has zero mean
        umean = sum(uwind,3)/real(nens)
        vmean = sum(vwind,3)/real(nens)
        do m=1,nens
            uwind(:,:,m) = uwind(:,:,m) - umean
            vwind(:,:,m) = vwind(:,:,m) - vmean
        end do

        synforc(:,:,:,uwind_id) = uwind
        synforc(:,:,:,vwind_id) = vwind


        !! Drag,  New drag - Computed directly from winds now
        if (taux_id .gt. 0) then
            wndfac=(1.+sign(1.,wndspd-11.))*.5
            cd_new=(0.49+0.065*wndspd)*1.0e-3*wndfac+cdfac*(1.-wndfac)
            synforc(:,:,:,taux_id) = uwind*wndspd*airdns*cd_new
            synforc(:,:,:,tauy_id) = vwind*wndspd*airdns*cd_new
        end if

    end if ! prsflg


end subroutine rand_update

end module module_random_field
