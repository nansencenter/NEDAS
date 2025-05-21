module qg_run_tools

  ! Contains core tools for running the model:
  ! PV inversion, time-stepping and forcing.
  !
  ! Routines: Get_pv, Invert_pv, March, Markovian, Write_snapshots
  !
  ! Dependencies: io_tools, qg_arrays, qg_params, qg_strat_tools, 
  !               transform_tools, numerics_lib
  
  implicit none
  save

contains

  !*******************************************************************

  function Get_pv(psi,surface_bc) result(q)

    !**************************************************************
    ! Calculate PV field from streamfunction
    !**************************************************************

    use op_rules,  only: operator(+), operator(-), operator(*)
    use qg_params, only: kmax, nz, nv, F
    use qg_arrays, only: psiq, ksqd_

    complex,dimension(-kmax:kmax,0:kmax,1-(nv-nz):nz),intent(in) :: psi
    character(*),intent(in)                                      :: surface_bc
    complex,dimension(-kmax:kmax,0:kmax,1:nz)                    :: q
    ! Local
    integer                                                      :: top
    
    select case (trim(surface_bc))
       case ('rigid_lid');    top = 1
       case ('surf_buoy');    top = 0
       case ('periodic');     top = nz
    end select

    if (nz>1) then
          q(:,:,1) = &
                 psiq(1,-1)*psi(:,:,top) &
               + psiq(1,0)*psi(:,:,1) &
               + psiq(1,1)*psi(:,:,2) 
          q(:,:,2:nz-1) = &
                 psiq(2:nz-1,-1)*psi(:,:,1:nz-2) &
               + psiq(2:nz-1,0)*psi(:,:,2:nz-1) &
               + psiq(2:nz-1,1)*psi(:,:,3:nz)
          q(:,:,nz) = &
                 psiq(nz,-1)*psi(:,:,nz-1) &
               + psiq(nz,0)*psi(:,:,nz) &
               + psiq(nz,1)*psi(:,:,1)
    else
       q = -F*psi
    endif

    q = -ksqd_*psi(:,:,1:nz) + q
    
  end function Get_pv

  !*******************************************************************

  function Invert_pv(surface_bc) result(psi)

    !**************************************************************
    ! Invert PV (q) in manner depending on surface_bc, which can be 
    ! either 'rigid_lid', 'free_surface', or 'periodic'.  These
    ! types are checked in qg_params/Check_parameters right now.
    !**************************************************************

    use op_rules,     only: operator(+), operator(-), operator(*)
    use qg_params,    only: kmax, nz, nv, F, nmask
    use qg_arrays,    only: psiq, ksqd_, lin2kx, lin2ky, q, b
    use numerics_lib, only: tridiag, tridiag_cyc

    complex,dimension(-kmax:kmax,0:kmax,1-(nv-nz):nz)  :: psi
    character(*),intent(in)                            :: surface_bc
    ! Local
    real,dimension(1-(nv-nz):nz,-1:1)                  :: psiq_m_ksqd
    complex,dimension(nv)                              :: qvec
    integer                                            :: kx, ky, kl

    psi = 0.
    psiq_m_ksqd = psiq

    if (nz>1) then               ! Multi-layer inversion

       select case (trim(surface_bc))
       case ('rigid_lid')

          do kl = 1,nmask
             kx = lin2kx(kl); ky = lin2ky(kl)
             psiq_m_ksqd(:,0) = psiq(:,0) - ksqd_(kx,ky) 
             psi(kx,ky,:) = tridiag(q(kx,ky,:),psiq_m_ksqd)
          enddo

       case ('surf_buoy')

          do kl = 1,nmask
             kx = lin2kx(kl); ky = lin2ky(kl)
             qvec(1) = b(kx,ky,1)
             qvec(2:nv) = q(kx,ky,1:nz)
             psiq_m_ksqd(1:nz,0) = psiq(1:nz,0) - ksqd_(kx,ky) 
             psi(kx,ky,:) = tridiag(qvec,psiq_m_ksqd)
          enddo
          
       case ('periodic')

          do kl = 1,nmask
             kx = lin2kx(kl); ky = lin2ky(kl)
             psiq_m_ksqd(:,0) = psiq(:,0) - ksqd_(kx,ky) 
             psi(kx,ky,:) = tridiag_cyc(q(kx,ky,:),psiq_m_ksqd)
          enddo          

       end select

    elseif (nz==1) then              ! Barotropic model inversion

       where (ksqd_/=0) psi(:,:,1) = -q(:,:,1)/(F+ksqd_)

    endif

  end function Invert_pv

  !*************************************************************************
  function Whitenoise(kf_min,kf_max,amp,field) result(noise)
    use numerics_lib, only: Ran
    use qg_arrays,    only: ksqd_
    use qg_params,    only: kmax,idum,i,pi
    integer :: nkx,nky
    real,intent(in) :: kf_min,kf_max,amp
    complex,dimension(:,:,:),intent(inout) :: field
    complex,dimension(size(field,1),size(field,2)) :: noise
    nkx=size(field,1); nky=size(field,2)
    noise=cmplx(0.0,0.0)
    where((ksqd_ > kf_min**2).and.(ksqd_ <= kf_max**2))
      noise=amp*cexp((i*2*pi)*Ran(idum,nkx,nky))
    endwhere
  end function Whitenoise

  !*************************************************************************
  function Markovian(kf_min,kf_max,amp,lambda,frc_o,norm_forcing,field) &
       result(forc)
    
    !**************************************************************
    ! Random Markovian forcing function.  If norm_forcing = T, function
    ! will normalize the forcing such that the total generation = amp
    !**************************************************************
    
    use op_rules,     only: operator(+), operator(-), operator(*)
    use qg_arrays,    only: ksqd_
    use qg_params,    only: kmax, idum, i, pi, rmf_norm_min
    use numerics_lib, only: Ran
    use io_tools,     only: Message

    real,intent(in)                                 :: kf_min,kf_max,amp,lambda
    complex,dimension(:,:),intent(inout)            :: frc_o
    logical,intent(in),optional                     :: norm_forcing
    complex,dimension(size(frc_o,1),size(frc_o,2)),intent(in),optional :: field
    complex,dimension(size(frc_o,1),size(frc_o,2))  :: forc
    ! Local
    real                                            :: gamma=1.
    integer                                         :: nkx, nky

    nkx = size(frc_o,1); nky = size(frc_o,2)

    if (present(norm_forcing)) then
       if ((norm_forcing).and.(.not.(present(field)))) then
          call Message('Error:Markovian: need field with norm_forcing',&
                        fatal='y')
       endif
    endif

    where((ksqd_ > kf_min**2).and.(ksqd_ <= kf_max**2))
       frc_o = lambda*frc_o &
             + amp*sqrt(1-lambda**2)*cexp((i*2*pi)*Ran(idum,nkx,nky))
    endwhere
    forc = frc_o 
    if (norm_forcing) then
       gamma = -2*sum(conjg(field)*forc)/amp
       if (abs(gamma)>rmf_norm_min) then
           forc = forc/gamma
       else
          call Message('RandMarkForc: normalization factor too small, =',&
                       r_tag = gamma)
       endif
    endif
    frc_o = forc
    
  end function Markovian

  !*********************************************************************

  function Write_snapshots(framein) result(frameout)

    !**************************************************************
    ! Write full snapshot of dynamic field and make restart files
    !**************************************************************

    use io_tools,    only: Message, Write_field
    use qg_params,   only: psi_file,tracer_file,time,time_file,&
                           use_tracer,use_forcing,use_forcing_t,&
                           force_o_file,force_ot_file,Write_parameters
    use qg_arrays,   only: psi,tracer,force_o,force_ot

    integer,intent(in) :: framein
    integer            :: frameout

    frameout = framein + 1                ! Update field frame counter

    call Write_field(psi,psi_file,frameout)       
    if (use_tracer)  then
       call Write_field(tracer,tracer_file,frameout) 
       if (use_forcing_t) call Write_field(force_ot,force_ot_file)
    endif
    if (use_forcing) call Write_field(force_o,force_o_file)
    call Write_parameters                 ! Write params for restart.nml
    call Write_field(time,time_file,frameout)  

    call Message('Wrote frame: ',tag=frameout)

  end function Write_snapshots

  !*********************************************************************

end module qg_run_tools
