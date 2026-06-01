module qg_arrays                !-*-f90-*-

  !************************************************************************
  ! This module contains all of the dynamically allocated field variables,
  ! and the routine to allocate memory for them.  Also sets the preset
  ! K^2 kx and ky grids.
  !
  ! Routines: Setup_fields
  !
  ! Dependencies: qg_params, io_tools
  !************************************************************************

  implicit none

  complex,dimension(:,:,:),allocatable   :: q,q_o,psi,psi_o,rhs
  complex,dimension(:,:,:),allocatable   :: b,b_o,rhs_b
  complex,dimension(:,:,:),allocatable   :: ug,vg,qxg,qyg,qdrag,unormbg
  complex,dimension(:,:),  allocatable   :: tracer,tracer_o,rhs_t,stir_field
  complex,dimension(:,:),  allocatable   :: hb,toposhift,force_o,force_ot
  real,   dimension(:,:),  allocatable   :: filter,filter_t
  real,   dimension(:,:,:),allocatable   :: tripint
  real,   dimension(:,:),  allocatable   :: ksqd_,kx_,ky_,vmode,psiq
  real,   dimension(:),    allocatable   :: ubar, vbar, um, vm, dz, rho, drho, kz
  real,   dimension(:),    allocatable   :: shearu, shearv, qbarx, qbary
  integer,dimension(:),    allocatable   :: kxv, kyv, lin2kx, lin2ky

  save

contains

  subroutine Setup_fields

    use io_tools,  only: Message, Write_field
    use qg_params, only: kmax,nz,use_tracer,use_forcing,use_forcing_t,&
                         use_topo,surf_buoy,nmask,quad_drag,nx,ny,z_stir

    real    :: filtdec
    integer :: kx, ky, n

    allocate(q(-kmax:kmax,0:kmax,1:nz),q_o(-kmax:kmax,0:kmax,1:nz))
    allocate(rhs(-kmax:kmax,0:kmax,1:nz))
    allocate(ug(nx,ny,nz),vg(nx,ny,nz),qxg(nx,ny,nz),qyg(nx,ny,nz))
    allocate(filter(-kmax:kmax,0:kmax))
    allocate(ksqd_(-kmax:kmax,0:kmax),kx_(-kmax:kmax,0:kmax))
    allocate(ky_(-kmax:kmax,0:kmax))
    allocate(kxv(2*kmax+1),kyv(kmax+1),kz(1:nz))
    allocate(rho(nz),dz(nz))
    allocate(ubar(nz),vbar(nz),um(nz),vm(nz),shearu(nz),shearv(nz))
    allocate(qbarx(nz),qbary(nz))
    allocate(vmode(1:nz,1:nz),tripint(nz,nz,nz))
    q=0.; q_o=0.; rhs=0.; rho=0; dz=0; vmode=1.; kz=0.; tripint=0.
    ubar=0.; vbar=0.; shearu=0.; shearv=0.; qbarx=0.; qbary=0. 
    if (nz==1) then
       dz=1.; rho=1.; ubar=0.;  vbar=0.
    endif

    ! Conditionally allocate special fields

    if (surf_buoy) then
       allocate(psi(-kmax:kmax,0:kmax,0:nz),psi_o(-kmax:kmax,0:kmax,0:nz))
       allocate(b(-kmax:kmax,0:kmax,1),b_o(-kmax:kmax,0:kmax,1))
       allocate(rhs_b(-kmax:kmax,0:kmax,1),psiq(0:nz,-1:1),drho(0:nz-1))
       b = 0.; b_o = 0.; rhs_b = 0.
    else
       allocate(psi(-kmax:kmax,0:kmax,1:nz),psi_o(-kmax:kmax,0:kmax,1:nz))
       allocate(psiq(1:nz,-1:1),drho(1:nz-1))
    endif
    psi = 0.; psi_o = 0.; psiq = 0.; drho=0.
    if (use_topo) then
       allocate(hb(-kmax:kmax,0:kmax),toposhift(-kmax:kmax,0:kmax))
       hb = 0.; toposhift=0.
    endif
    if (use_tracer) then
       allocate(tracer(-kmax:kmax,0:kmax))
       allocate(tracer_o(-kmax:kmax,0:kmax))
       allocate(rhs_t(-kmax:kmax,0:kmax))
       allocate(filter_t(-kmax:kmax,0:kmax))
       allocate(stir_field(-kmax:kmax,0:kmax))
       tracer = 0.; tracer_o = 0.; rhs_t = 0.; filter_t = 0.; stir_field = 0.
       if (use_forcing_t) then
          allocate(force_ot(-kmax:kmax,0:kmax))
          force_ot = 0.
       endif
    endif
    if (use_forcing) then
       allocate(force_o(-kmax:kmax,0:kmax))
       force_o = 0.
    endif
    if (quad_drag/=0) then
       allocate(qdrag(-kmax:kmax,0:kmax,1),unormbg(nx,ny,1))
       qdrag = 0.; unormbg = 0.
    endif

    ! Store values of kx, ky and k^2 in 2d arrays.

    kxv = (/ (kx,kx=-kmax,kmax) /)
    kyv = (/ (ky,ky=0,kmax) /)
    kx_ = float(spread(kxv,2,kmax+1))
    ky_ = float(spread(kyv,1,2*kmax+1))
    ksqd_ = kx_**2 + ky_**2
    ksqd_(0,0) = .1    ! 0,0 never used - this way can divide by K^2 w/o worry

    ! Get # of true elements in de-aliasing mask and set up index maps so 
    ! that we can choose only to invert dynamic part of psi

    filter = 1.                ! Set up de-aliasing part of filter first
    where (ksqd_>= (8./9.)*(kmax+1)**2) filter = 0.
    filter(-kmax:0,0) = 0.
    nmask = sum(int(filter))   ! Get non-zero total of filter 

    allocate(lin2kx(nmask),lin2ky(nmask))
    lin2kx = pack(kx_,MASK=(filter>0))
    lin2ky = pack(ky_,MASK=(filter>0))

  end subroutine Setup_fields

end module qg_arrays
