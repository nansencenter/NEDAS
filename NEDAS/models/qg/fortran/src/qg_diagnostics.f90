module qg_diagnostics

  !************************************************************************
  ! Energetics and other diagnostics
  ! 
  ! Routines:  Get_energetics, Get_spectra
  !
  ! Dependencies: qg_arrays, qg_params, io_tools, op_rules, transform_tools
  !
  !************************************************************************

  implicit none
  private
  save

  public :: Get_energetics, Get_spectra, energy, enstrophy, zsq

contains

  !*************************************************************************

  real function energy(psi)

    ! Get total energy
    
    use op_rules,  only: operator(*)
    use qg_params, only: nz, nv, kmax, F, Fe
    use qg_arrays, only: dz, ksqd_, drho

    complex,dimension(-kmax:kmax,0:kmax,1-(nv-nz):nz),intent(in) :: psi
    real                                :: ke=0., ape=0.

    ke = sum(ksqd_*dz*psi(:,:,1:nz)*conjg(psi(:,:,1:nz)))
    if (nz>1) then
       ape = sum(F*conjg(psi(:,:,2:nz)-psi(:,:,1:nz-1))* &
            (psi(:,:,2:nz)-psi(:,:,1:nz-1))*(1/drho(1:nz-1))) &
            + Fe*sum(conjg(psi(:,:,1))*psi(:,:,1))
    else
       ape = F*sum(psi(:,:,1:nz)*conjg(psi(:,:,1:nz)))
    endif
    
    if (nv-nz==1) then   !if (trim(surface_bc)=='surf_buoy') then
       ke = ke + sum(ksqd_*dz(1)*psi(:,:,0)*conjg(psi(:,:,0)))
       ape = ape + sum(F*conjg(psi(:,:,1)-psi(:,:,0))* &
            (psi(:,:,1)-psi(:,:,0))*(1/drho(1)))
    endif

    energy = ke + ape

  end function energy

  !*************************************************************************

  real function enstrophy(q)

    ! Get total enstrophy
    
    use op_rules,  only: operator(*)
    use qg_arrays, only: dz
    
    complex,dimension(:,:,:),intent(in) :: q

    enstrophy = sum( dz*q*conjg(q) )

  end function enstrophy

  !*************************************************************************

  real function zsq(psi)

    ! Get mean square vorticity

    use op_rules,  only: operator(*)
    use qg_params, only: nz, nv, kmax
    use qg_arrays, only: dz,ksqd_
    
    complex,dimension(-kmax:kmax,0:kmax,1-(nv-nz):nz),intent(in) :: psi

    zsq = 2*sum( dz*(ksqd_**2*psi(:,:,1:nz)*conjg(psi(:,:,1:nz))))

  end function zsq

  !*************************************************************************

  function Get_energetics(framein) result(dframe)

    !************************************************************************
    ! Calculate KE, APE, and ENS, as well as all rates of generation
    ! and dissipation.  Also calculates current eddy rotation period.
    ! All energetics factors multiplied by 2 since we are only using
    ! upper-half plane spectral fields
    !************************************************************************
    
    use op_rules,  only: operator(+), operator(-), operator(*)
    use io_tools,  only: Message, Write_field
    use qg_arrays, only: ksqd_,kx_,ky_,dz,psi,psi_o,q,shearu,shearv,ubar,vbar,force_o,drho, &
                         tracer,b,qdrag,filter,filter_t,force_ot,stir_field
    use qg_params, only: dt,nz,kmax,bot_drag,top_drag,i,pi,F,Fe,therm_drag,&
                         time,cntr,uscale,vscale,use_forcing,use_tracer,&
                         quad_drag,surface_bc,filter_type,filter_type_t, &
                         use_forcing_t,use_mean_grad_t

    integer,intent(in) :: framein
    integer            :: dframe
    real,dimension(:,:),allocatable  :: temp
    real               :: ke=0., ens=0.,ape=0.,dedt=0.
    real               :: gen_bci_rate=0., gen_rmf_rate=0., thd_rate=0.
    real               :: bd_rate=0.,qd_rate=0.,filter_rate=0.,filter_rate_t=0.
    real               :: td_rate=0.,eddy_time=0.,tvar=0.,tpsi=0.,zeta_rms=0.
    real               :: gen_rmf_t=0., gen_tg_t=0.

    dframe = framein + 1        ! Update diagnostics frame counter
    call Write_field(time,'diag1_time',dframe) ! Track diagnostic-writes
    
    ke =  sum(ksqd_*dz*psi(:,:,1:nz)*conjg(psi(:,:,1:nz)))
    call Write_field(ke,'ke',dframe)

    dedt = (ke - sum(ksqd_*dz*psi_o(:,:,1:nz)*conjg(psi_o(:,:,1:nz))))/dt
    call Write_field(dedt,'dedt',dframe)

    ens = sum( dz*(q*conjg(q)) )
    call Write_field(ens,'ens',dframe) 

    if (trim(filter_type)/='none') then
       allocate(temp(-kmax:kmax,0:kmax)); temp = 0.
       where (filter/=0) temp = (filter**(-1)-1)/(2*dt)
       filter_rate = 2*sum(real( temp*(dz*(conjg(psi(:,:,1:nz))*q)) ))
       if (trim(surface_bc)=='surf_buoy') then
          filter_rate = filter_rate + dz(1)*F/drho(0) &
               *sum(real( temp *conjg(psi(:,:,1))*b ))
       endif
       deallocate(temp)
       call Write_field(filter_rate,'filter_rate',dframe)
    endif
    if (bot_drag/=0) then
       bd_rate = -2*sum(bot_drag*ksqd_*dz(nz)*conjg(psi(:,:,nz))*psi(:,:,nz))
       call Write_field(bd_rate,'bd_rate',dframe)
    endif
    if (quad_drag/=0) then
       qd_rate = 2*sum(dz(nz)*conjg(psi(:,:,nz))*qdrag)
       call Write_field(qd_rate,'qd_rate',dframe)
    endif
    if (top_drag/=0) then
       td_rate = -2*sum(top_drag*ksqd_*dz(1)*conjg(psi(:,:,1))*psi(:,:,1))
       call Write_field(td_rate,'td_rate',dframe)
    endif
    if (use_forcing) then
       gen_rmf_rate = - 2*sum(real(conjg(psi(:,:,1:nz))*force_o))
       call Write_field(gen_rmf_rate,'gen_rmf_rate',dframe)
    endif
    if ((trim(surface_bc)=='surf_buoy').and.therm_drag/=0) then
       thd_rate = dz(1)*F/drho(0)*therm_drag*sum(conjg(psi(:,:,1))*b)
       call Write_field(thd_rate,'thd_rate',dframe)
    endif
    if (nz>1) then
       ape = sum(F*conjg(psi(:,:,2:nz)-psi(:,:,1:nz-1))* &
            (psi(:,:,2:nz)-psi(:,:,1:nz-1))*(1/drho(1:nz-1))) &
            + Fe*sum(conjg(psi(:,:,1))*psi(:,:,1))
       call Write_field(ape,'ape',dframe)
       gen_bci_rate = -2*sum(real(i*(kx_*(dz*conjg(psi(:,:,1:nz))&
                  *(shearu(1:nz)*psi(:,:,1:nz)-ubar(1:nz)*q)))))&
                      -2*sum(real(i*(ky_*(dz*conjg(psi(:,:,1:nz))&
                  *(shearv(1:nz)*psi(:,:,1:nz)-vbar(1:nz)*q)))))
       call Write_field(gen_bci_rate,'gen_bci_rate',dframe)
       if (therm_drag/=0) then
          if (Fe/=0) then
             thd_rate = 2*sum(therm_drag*conjg(psi)*(q+ksqd_*psi(:,:,1:nz)))
          elseif (Fe==0) then
             thd_rate = 2*F*sum(real(therm_drag* &
                  (-conjg(psi(:,:,1))*(psi(:,:,1)-psi(:,:,2)) &
                   -conjg(psi(:,:,2))*(psi(:,:,2)-psi(:,:,1)))))
          endif
          call Write_field(thd_rate,'thd_rate',dframe)
       endif
    elseif (nz==1) then
       if (F/=0) then
          ape = F*sum(psi*conjg(psi))
          call Write_field(ape,'ape',dframe)
          if (therm_drag/=0) then
             thd_rate = -2*sum(therm_drag*F*conjg(psi)*psi)
             call Write_field(thd_rate,'thd_rate',dframe)
          endif
       endif
    endif

    if (use_tracer) then
       tvar = sum(tracer*conjg(tracer))
       call Write_field(tvar,'tvar',dframe)
       tpsi = sum(tracer*conjg(stir_field))
       call Write_field(tpsi,'tpsi',dframe)
       if (trim(filter_type_t)/='none') then
          allocate(temp(-kmax:kmax,0:kmax)); temp = 0.
          where (filter_t/=0) temp = (filter_t**(-1)-1)/(2*dt)
          filter_rate_t = -2*sum(real( temp*conjg(tracer)*tracer ))
          call Write_field(filter_rate_t,'filter_rate_t',dframe)
          deallocate(temp)
       endif
       if (use_mean_grad_t) then
          gen_tg_t = - 2*sum(real(tracer*conjg(i*kx_*stir_field)))
          call Write_field(gen_tg_t,'gen_tg_t',dframe)             
       endif
       if (use_forcing_t) then
          gen_rmf_t = - 2*sum(real(conjg(tracer)*force_ot))
          call Write_field(gen_rmf_t,'gen_rmf_t',dframe)
       endif
    endif

    zeta_rms = sqrt(2*sum( dz*(ksqd_**2*psi(:,:,1:nz)*conjg(psi(:,:,1:nz))) ))
    eddy_time = 2*pi/zeta_rms
    call Write_field(eddy_time,'eddy_time',dframe)
    
    if (.not.ieee_is_finite(ke)) call Message('INF or NAN in ke - quitting!',fatal='y')

    ! Write some information to screen and to log file

    call Message('')          
    call Message('time step     =',tag=cntr)
    call Message('energy        =',r_tag=ke+ape)
    call Message('enstrophy     =',r_tag=ens)
    if (nz>1.and.(uscale/=0..or.vscale/=0)) &
                      call Message('gen_bci_rate  =',r_tag=gen_bci_rate)
    if (use_forcing) call Message('gen_rmf_rate  =',r_tag=gen_rmf_rate)
    if (bot_drag/=0)  call Message('botdrag_rate  =',r_tag=bd_rate)
    if (quad_drag/=0) call Message('quaddrag_rate =',r_tag=qd_rate)
    if (therm_drag/=0)call Message('thermdrag_rate=',r_tag=thd_rate)
    if (trim(filter_type)/='none') &
                      call Message('filter_rate   =',r_tag=filter_rate)  ! temp
    if (use_tracer) then
       call Message('tvariance     =',r_tag=tvar)
       call Message('tvar_gen      =',r_tag=gen_rmf_t+gen_tg_t)
       call Message('tvar_dissip   =',r_tag=filter_rate_t)
    endif

  end function Get_energetics

  !*************************************************************************

  function Get_spectra(framein) result(dframe)

    !************************************************************************
    ! Calculate the isotropic horizontal wavenumber vs vertical
    ! wavenumber spectra of modal and layered energetics
    !************************************************************************

    use op_rules,        only: operator(+), operator(-), operator(*)
    use io_tools,        only: Message, Write_field
    use transform_tools, only: Spec2grid, Jacob
    use numerics_lib,    only: Ring_integral, sub2ind
    use strat_tools,     only: Layer2mode
    use qg_arrays,       only: ksqd_,kx_,ky_,kxv,kyv,dz,psi,psi_o,q,shearu,shearv,&
                               ubar,vbar,filter,force_o,drho,tracer,vmode,kz, &
                               tripint,um,vm,qdrag,filter_t,stir_field
    use qg_params,       only: time,dt,nx,ny,kmax,nz,bot_drag,top_drag,&
                               i,pi,F,uscale,vscale,therm_drag,cntr,quad_drag,&
                               use_forcing,use_tracer,do_xfer_spectra,&
                               do_x_avgs,do_genm_spectra,filter_type,&
                               filter_type_t,use_mean_grad_t,do_aniso_spectra,&
                               calc_residual,Fe

    integer,intent(in)                     :: framein
    integer                                :: dframe
    real,dimension(:,:,:),allocatable      :: field
    real,dimension(:,:),allocatable        :: spec, spec_m3, xavg, temp
    real,dimension(:),allocatable          :: vq 
    complex,dimension(:,:,:),allocatable   :: psim
    integer                                :: m, j, k
    logical,save                           :: called_yet=.false.

    if (.not.called_yet) then
       if (do_xfer_spectra) call Message('Will calculate modal transfer &
                                  &spectra - very expensive for nz>2.')
       if (do_genm_spectra) call Message('Will calculate modal generation &
                                  &spectra - very expensive for nz>2.')
       called_yet=.true.
    endif

    dframe = framein + 1
    call Write_field(time,'diag2_time',dframe) ! Track diagnostic-writes
    
    allocate(spec(kmax,nz),field(-kmax:kmax,0:kmax,nz))
    allocate(xavg(ny,nz),spec_m3(kmax,nz**3),vq(nz))

    ! KE spectra

    field = dz*(ksqd_*psi(:,:,1:nz)*conjg(psi(:,:,1:nz)))
    spec = Ring_integral(real(field),kxv,kyv,kmax)
    call Write_field(spec,'kes',dframe)

    ! Spectra of energy along kx and ky axes if anisotropy expected
    
    if (do_aniso_spectra) then
       spec = real(field(1:kmax,0,:))
       call Write_field(spec,'kesx',dframe)
       spec = real(field(0,1:kmax,:))
       call Write_field(spec,'kesy',dframe)
    endif
    
    if (calc_residual) then
       field = (field - ksqd_*dz*psi_o(:,:,1:nz)*conjg(psi_o(:,:,1:nz)) )/dt
       call Write_field(field,'dedtf',dframe)       
       spec = Ring_integral(real(field),kxv,kyv,kmax)
       call Write_field(spec,'dedts',dframe)
    endif

    if (uscale/=0.or.vscale/=0) then               ! From mean shear forcing
       field = -real(2*i*(dz(1:nz)*(shearu(1:nz)*psi(:,:,1:nz)-ubar(1:nz)*q) &
            *(kx_*conjg(psi(:,:,1:nz))))) &
             -real(2*i*(dz(1:nz)*(shearv(1:nz)*psi(:,:,1:nz)-vbar(1:nz)*q) &
            *(ky_*conjg(psi(:,:,1:nz))))) 
       spec = Ring_integral(2*real(field),kxv,kyv,kmax)
       call Write_field(spec,'gens',dframe)
    endif
    if (use_forcing) then        ! From random Markovian forcing
       field(:,:,1) = -conjg(psi(:,:,1))*force_o
       if (nz>1) field(:,:,2:nz) = 0.
       spec = Ring_integral(2*real(field),kxv,kyv,kmax)
       call Write_field(spec,'gens_rmf',dframe)
    endif

    ! Calculate BC diagnostics

    multilayer: if (nz>1) then  

       allocate(psim(-kmax:kmax,0:kmax,1:nz))
       psim = Layer2mode(psi(:,:,1:nz),vmode,dz)

       field = ksqd_*psim*conjg(psim)   ! Modal KE
       spec = Ring_integral(real(field),kxv,kyv,kmax)
       call Write_field(spec,'kems',dframe)

       ! Spectra of energy along kx and ky axes if anisotropy expected

       if (do_aniso_spectra) then
          spec = real(field(1:kmax,0,:))
          call Write_field(spec,'kemsx',dframe)
          spec = real(field(0,1:kmax,:))
          call Write_field(spec,'kemsy',dframe)
       endif
       
       field = kz**2*psim*conjg(psim)         ! Modal APE
       spec = Ring_integral(real(field),kxv,kyv,kmax)
       call Write_field(spec,'apems',dframe)

       field(:,:,1)    = Fe*conjg(psi(:,:,1))*psi(:,:,1)  ! Interface APE
       field(:,:,2:nz) = F*conjg(psi(:,:,2:nz)-psi(:,:,1:nz-1))* &
            (psi(:,:,2:nz)-psi(:,:,1:nz-1))*(1/drho(1:nz-1))
       spec = Ring_integral(real(field(:,:,1:nz)),kxv,kyv,kmax)
       call Write_field(spec,'apes',dframe)

       if (do_genm_spectra) then
          if (uscale/=0) then     ! Modal eddy generation -- !!vbar not yet included!!
             do m = 1,nz
                do j = 1,nz
                   do k = 1,nz
                      field(:,:,1) =  tripint(j,k,m)*um(j)* &
                           conjg(psim(:,:,m))*i*kx_* &
                           (kz(j)**2-kz(k)**2-ksqd_)*psim(:,:,k)
                      spec_m3(:,sub2ind((/ k,j,m /),nz)) = &
                           Ring_integral(2*real(field(:,:,1)),kxv,kyv,kmax)
                   enddo
                enddo
             enddo
             call write_field(spec_m3,'genms',dframe)
          endif
       endif

       if (therm_drag/=0) then   ! Modal thermal drag dissipation 
          do m = 1,nz
             field(:,:,m) = -therm_drag*F/drho(1)*sum((vmode(1,:)-vmode(2,:))*psim,3) &
                  *(conjg(psim(:,:,m))*(vmode(1,m)-vmode(2,m)))
          enddo
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          call Write_field(spec,'thdms',dframe)
       endif
          
       if (bot_drag/=0) then     ! Modal bottom drag dissipation 
          do m = 1,nz
             field(:,:,m) = -bot_drag*dz(nz)*ksqd_*sum(vmode(nz,:)*psim,3) &
                  *(vmode(nz,m)*conjg(psim(:,:,m))) 
          enddo
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          call Write_field(spec,'bdms',dframe)
       endif
    
       if (top_drag/=0) then     ! Modal top drag dissipation 
          do m = 1,nz
             field(:,:,m) = -top_drag*dz(1)*ksqd_*sum(vmode(1,:)*psim,3)  &
                            *(vmode(1,m)*conjg(psim(:,:,m)))
          enddo
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          call Write_field(spec,'tdms',dframe)
       endif

       if (quad_drag/=0) then     ! Modal quadratic drag dissipation 
          do m = 1,nz
             field(:,:,m) = -dz(nz)*sum(vmode(nz,:)*qdrag,3) &
                  *(vmode(nz,m)*conjg(psim(:,:,m))) 
          enddo
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          call Write_field(spec,'qdms',dframe)
       endif
    
       if (trim(filter_type)/='none') then   ! Modal filter dissipation
          allocate(temp(-kmax:kmax,0:kmax)); temp = 0.
          where (filter/=0) temp = (filter**(-1)-1)/(2*dt)
          field = -(temp*(ksqd_+kz**2))*psim*conjg(psim)
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          deallocate(temp)
          call Write_field(spec,'filterms',dframe)        
       endif

       if (do_xfer_spectra) then         ! Internal transfer terms
          do m = 1,nz           
             do j = 1,nz
                do k = 1,nz
                   if (abs(tripint(m,j,k))>1000*epsilon(tripint(m,j,k))) then

                      field(:,:,1) = tripint(m,j,k)*conjg(psim(:,:,m)) &
                           *jacob(psim(:,:,j),-(kz(k)**2+ksqd_)*psim(:,:,k))
                      spec_m3(:,sub2ind((/ k,j,m /),nz)) = &
                           Ring_integral(2*real(field(:,:,1)),kxv,kyv,kmax)

                   else
                      spec_m3(:,sub2ind((/ k,j,m /),nz)) = 0.
                   endif
                enddo
             enddo
          enddo
          call Write_field(spec_m3,'xferms',dframe)
       endif
       deallocate(psim)

       ! Area averaged eddy PV flux as function of depth

       vq = 2*sum(sum(real((i*kx_*psi(:,:,1:nz))*conjg(q)),1),1)
       call write_field(vq,'vq',dframe)

    elseif (nz==1) then
    
       if (F/=0) then             ! Calculate APE spectrum
          field = F*real(conjg(psi)*psi)
          spec = Ring_integral(real(field),kxv,kyv,kmax)
          call Write_field(spec,'apes',dframe)
       endif

       if (do_xfer_spectra) then
          field = real(conjg(psi)*jacob(psi,q))
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          call Write_field(spec,'xfers',dframe)
       endif

       if (bot_drag/=0) then     ! Bottom drag dissipation 
          field = -bot_drag*ksqd_*psi(:,:,1:nz)*conjg(psi(:,:,1:nz)) 
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          call Write_field(spec,'bds',dframe)
       endif
       
       if (trim(filter_type)/='none') then    ! Filter dissipation
          allocate(temp(-kmax:kmax,0:kmax)); temp = 0.
          where (filter/=0) temp = (filter**(-1)-1)/(2*dt)
          field = -temp*(ksqd_*psi(:,:,1:nz)*conjg(psi(:,:,1:nz)))
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          deallocate(temp)
          call Write_field(spec,'filters',dframe)        
       endif

       if (quad_drag/=0) then      ! Quadratic drag dissipation
          field = (dz(nz)*conjg(psi(:,:,nz))*qdrag)
          spec = Ring_integral(2*real(field),kxv,kyv,kmax)
          call Write_field(spec,'qds',dframe)
       endif

    endif multilayer

    ! Write tracer spectra

    if (use_tracer) then
       field(:,:,1) = real(tracer*conjg(tracer))      ! tracer variance <t't'>
       spec = Ring_integral(field,kxv,kyv,kmax)
       call Write_field(spec(:,1),'tvars',dframe)
       field(:,:,1) = -real(tracer*conjg(stir_field)) ! tracer psi cor <t'psi'>
       spec = Ring_integral(field,kxv,kyv,kmax)
       call Write_field(spec(:,1),'tpsis',dframe)
       if (use_mean_grad_t) then
          field(:,:,1) = -real(tracer*conjg(i*kx_*stir_field)) ! <t'v'> vs. k
          spec = Ring_integral(2*field,kxv,kyv,kmax)
          call Write_field(spec(:,1),'tfluxs',dframe)
       endif
    endif

    ! Calculate zonal average momentum and PV fluxes

    if (do_x_avgs) then    ! Zonal averages 
       xavg = sum(Spec2grid(-i*(ky_*psi(:,:,1:nz)))&
              *Spec2grid(i*(kx_*psi(:,:,1:nz))),2)/nx ! <u'v'>
       call Write_field(xavg,'uv_avg_x',dframe)
       xavg = sum(Spec2grid(i*kx_*psi(:,:,1:nz))*Spec2grid(q),2)/nx  ! <v'q'>
       call Write_field(xavg,'vq_avg_x',dframe)
    endif

    deallocate(field,spec,spec_m3,xavg,vq)

  end function Get_spectra

  !*********************************************************************

   elemental logical function ieee_is_nan(x)
     real,intent(in):: x
     ieee_is_nan = isnan(x)
   end function ieee_is_nan
   elemental logical function ieee_is_finite(x)
     real,intent(in):: x
     ieee_is_finite = .not. (isnan(x) .or. abs(x) > huge(x))
   end function ieee_is_finite

  !*********************************************************************

end module qg_diagnostics
