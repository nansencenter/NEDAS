program qg_driver 

  ! Stratified or barotropic spectral, homogeneous QG model
  !
  ! Routines: main, Get_rhs, Get_rhs_t
  !
  ! Dependencies: everything.

  use op_rules,        only: operator(+), operator(-), operator(*)
  use qg_arrays,       only: q,q_o,psi,psi_o,rhs,b,b_o,rhs_b,tracer,tracer_o, &
                             rhs_t,force_o,force_ot,filter,filter_t,hb,toposhift, &
                             qxg,qyg,ug,vg,unormbg,qdrag,ubar,vbar,drho,qbarx,qbary,&
                             shearu,shearv,um,vm,&
                             kx_,ky_,ksqd_,psiq,vmode,dz,rho,stir_field, &
                             Setup_fields
  use qg_params,       only: kmax,nz,filter_type,filter_exp,k_cut,strat_type, &
                             surface_bc,ubar_type,vbar_type,beta,F,Fe,use_topo,topo_type, &
                             restarting,psi_init_type,surf_buoy,use_tracer,&
                             cr,ci,filter_type_t,filter_exp_t,k_cut_t,e_o, &
                             tracer_init_type,parameters_ok,cntr,cnt, &
                             total_counts,start,diag1_step,diag2_step,time, &
                             d1frame,d2frame,frame,do_spectra,write_step, &
                             adapt_dt,dt_step,dt_tune,pi,i,nx,dt,robert, &
                             call_q,call_b,call_t,bot_drag,top_drag,z_stir, &
                             therm_drag,kf_min,kf_max,forc_coef,forc_corr, &
                             norm_forcing,use_mean_grad_t,use_forcing_t, &
                             kf_min_t,kf_max_t,forc_coef_t,norm_forcing_t, &
                             linear,quad_drag,qd_angle,use_forcing,uscale,vscale,umode, &
                             dealiasing,dealiasing_t,filt_tune,filt_tune_t, &
                             force_o_file,force_ot_file, &
                             ubar_in_file,ubar_file,vbar_in_file,vbar_file, &
                             Initialize_parameters
  use qg_run_tools,    only: Get_pv,Invert_pv,Markovian,Whitenoise,Write_snapshots
  use qg_init_tools,   only: Init_strat, Init_ubar, Init_topo, Init_tracer, &
                             Init_streamfunction, Init_filter, Init_counters, &
                             Init_forcing
  use qg_diagnostics,  only: Get_energetics, Get_spectra, energy, zsq
  use transform_tools, only: Init_transform, Spec2grid_cc, Grid2spec, Jacob, &
                             ir_prod, ir_pwr
  use numerics_lib,    only: Tri2mat, March, Ran
  use io_tools,        only: Message, write_field

  implicit none

  ! *********** Model initialization *********************

  call Initialize_parameters           ! Read/set/check params (in qg_params)

  call Message('Initializing model...')

  call Init_counters                   ! (in qg_init_tools)
  call Init_transform(kmax,nz)         ! Init transform (in transform_tools)
  call Setup_fields                    ! Allocate/init fields (in qg_arrays)

  filter = Init_filter(filter_type,filter_exp,k_cut,dealiasing,filt_tune,1.)

  if (nz>1) then                       ! Read/create mean profiles
     call Init_strat(strat_type,surface_bc) ! Set dz,rho,vmode,kz,tripint,psiq 
     if (uscale/=0.and.ubar_type/='none') then
        ubar = Init_ubar(ubar_type,ubar_in_file,uscale,umode,ubar_file)
        shearu = matmul(tri2mat(psiq(1:nz,-1:1)),ubar)
        um = matmul(transpose(vmode),dz*ubar(1:nz))    ! Project shear onto modes
     endif
     if (vscale/=0.and.vbar_type/='none') then
        vbar = Init_ubar(vbar_type,vbar_in_file,vscale,umode,vbar_file)
        shearv = matmul(tri2mat(psiq(1:nz,-1:1)),vbar)
        vm = matmul(transpose(vmode),dz*vbar(1:nz))    ! Project shear onto modes
     endif
  else
     ubar = uscale
     vbar = vscale
  endif
  qbarx = shearv(1:nz)
  qbary = -shearu(1:nz) + beta            ! Mean PV gradient
  if (use_topo) then
     hb = Init_topo(topo_type,restarting) ! Read/create bottom topo
     toposhift = - i*ubar(nz)*(kx_*hb)- i*vbar(nz)*(ky_*hb)
  endif
  if (use_forcing) force_o = Init_forcing(forc_coef,forc_corr,kf_min,kf_max,&
                                          norm_forcing,force_o_file,restarting)

  psi = Init_streamfunction(psi_init_type,restarting)  ! Read/create psi

  !!!!!MY: add random model error
  !psi = psi + spread(Whitenoise(kf_min,kf_max,forc_coef,psi),3,nz)

  q = Get_pv(psi,surface_bc)                           ! Get initial PV 

  if (surf_buoy) b(:,:,1) = (psi(:,:,0)-psi(:,:,1))/dz(1) ! d(psi)/dz(0)

  call Get_rhs                                         ! Get initial RHS

  if (use_tracer) then
     call Message('Tracers on')
     call Message('Any following messages regarding filter parameters')
     call Message('  refer to tracer filter: append _t to variable names')
     filter_t = Init_filter(filter_type_t,filter_exp_t,k_cut_t,dealiasing_t, &
                            filt_tune_t,1.)
     if (use_forcing_t) force_ot = Init_forcing(forc_coef_t,forc_corr, &
                                                kf_min_t,kf_max_t,&
                                                norm_forcing_t,force_ot_file,&
                                                restarting)
     tracer = Init_tracer(tracer_init_type,restarting) ! Init tracer
     call Get_rhs_t  ! Bottom of this file
  endif

  if (.not.parameters_ok) then
     call Message('The listed errors pertain to values set in your input')
     call Message('namelist file - correct the entries and try again.', &
                   fatal='y')
  endif
     
  ! *********** Main time loop *************************

  call Message('Beginning calculation')

  do cntr = cnt, total_counts           ! Main time loop

     start = (cntr==1)                  ! Flag true if 1st step of run

     ! Calculate diagnostics, write output
     if (mod(cntr,diag1_step)==0.or.start) d1frame = Get_energetics(d1frame)
     if (do_spectra.and.(mod(cntr,diag2_step)==0.or.start)) &
                                           d2frame = Get_spectra(d2frame)
     if (mod(cntr,write_step)==0.or.start) frame = Write_snapshots(frame)

     !!!!!!MY: output file
     if (cntr==total_counts) &
       call Write_field(psi,'output',1)

     if (adapt_dt.and.(mod(cntr,dt_step)==0.or.start)) &    ! Adapt dt
          dt = dt_tune*2*pi/(kmax*sqrt(max(zsq(psi),beta,1.)))

     q = filter*March(q,q_o,rhs,dt,robert,call_q)
     if (surf_buoy) b = filter*March(b,b_o,rhs_b,dt,robert,call_b)

     psi_o = psi                        ! Save for time lagged dissipation
     psi = Invert_pv(surface_bc)        ! Invert q to get psi

     call Get_rhs                       ! See below

     if (use_tracer) then 
        tracer = filter_t*March(tracer,tracer_o,rhs_t,dt,robert,call_t)
        call Get_rhs_t                  ! See below
     endif

     time = time + dt                   ! Update clock

  enddo                                 ! End of main time loop

  call Message('Calculation done')

!*********************************************************************

contains

  ! Separate RHS calculations so that they can be conveniently
  ! called for initialization and in loop 

  !*********************************************************************

  subroutine Get_rhs

     ! Get physical space velocity and pv gradient terms for
     ! use in calculation of advection (jacobian) and quadratic drag

     if (.not.linear) then
        
        ug  = Spec2grid_cc(-i*ky_*psi)      ! Calculate derivatives and
        vg  = Spec2grid_cc(i*kx_*psi)       ! transform to grid space --
        if (use_topo) q(:,:,nz) = q(:,:,nz) + hb
        qxg = Spec2grid_cc(i*kx_*q) 
        qyg = Spec2grid_cc(i*ky_*q)
        if (use_topo) q(:,:,nz) = q(:,:,nz) - hb
           
        ! Now do product in grid space and take difference back to k-space
        ! (ir_prod separately multiplies re and im parts of field, keeping
        ! staggered and straight grid forms in place)

        rhs = -Grid2spec(ir_prod(ug,qxg) + ir_prod(vg,qyg)) 

        ! Quadratic drag on bottom layer (ir_pwr raises field to power via
        ! method of ir_prod)

        if (quad_drag/=0) then  
           unormbg(:,:,1) = ir_pwr(ir_pwr(ug(:,:,nz),2.) &
                                 + ir_pwr(vg(:,:,nz),2.),.5)
           qdrag = quad_drag*filter &
             *( i*kx_*Grid2spec(unormbg* &
                (ug(:,:,nz)*sin(qd_angle)+vg(:,:,nz)*cos(qd_angle))) &
               -i*ky_*Grid2spec(unormbg* &
                (ug(:,:,nz)*cos(qd_angle)-vg(:,:,nz)*sin(qd_angle))))
           rhs(:,:,nz) = rhs(:,:,nz) - qdrag(:,:,1)
        endif

     endif

     ! Linear forcing and dissipation terms

     if (any(qbarx/=0)) rhs = rhs - i*(ky_*(-qbarx*psi(:,:,1:nz) + vbar*q(:,:,1:nz)))
     if (any(qbary/=0)) rhs = rhs - i*(kx_*( qbary*psi(:,:,1:nz) + ubar*q(:,:,1:nz)))
     if (use_topo)    rhs(:,:,nz) = rhs(:,:,nz) + toposhift
     if (bot_drag/=0) rhs(:,:,nz) = rhs(:,:,nz) + bot_drag*ksqd_*psi_o(:,:,nz)
     if (top_drag/=0) rhs(:,:,1)  = rhs(:,:,1)  + top_drag*ksqd_*psi_o(:,:,1)
     if (use_forcing) rhs(:,:,1)  = rhs(:,:,1) + Markovian(kf_min,kf_max, &
                        forc_coef,forc_corr,force_o,norm_forcing,psi(:,:,1))
     if (surf_buoy) then
        rhs_b(:,:,1) = -Jacob((psi(:,:,0)+psi(:,:,1))/2,b(:,:,1))
        if (therm_drag/=0) rhs_b = rhs_b - therm_drag*b_o
     endif
     if (therm_drag/=0.and.F/=0) then
        if (nz==1) then
           rhs = rhs + therm_drag*F*psi_o
        elseif (nz>1.and.Fe/=0) then
           rhs = rhs - therm_drag*(q+ksqd_*psi(:,:,1:nz))
        elseif (nz>1.and.Fe==0) then
           rhs(:,:,1) = rhs(:,:,1) - therm_drag*F*(psi_o(:,:,2)-psi_o(:,:,1))/dz(1)
           rhs(:,:,2) = rhs(:,:,2) - therm_drag*F*(psi_o(:,:,1)-psi_o(:,:,2))/dz(2)
        endif
     endif

     rhs = filter*rhs

  end subroutine Get_rhs

  !*********************************************************************

  subroutine Get_rhs_t

    real :: ut

    if (z_stir==0) then             ! Stir tracer with BT streamfunction
       stir_field = sum(dz*psi,3)   ! BT psi
       ut = 0.
    elseif (z_stir>0.and.z_stir<=nz) then
       stir_field = psi(:,:,z_stir)
       ut = ubar(z_stir)
    endif

    rhs_t = -Jacob(stir_field,tracer) 
    if (use_mean_grad_t) rhs_t = rhs_t - i*kx_*stir_field
    if (ut/=0)           rhs_t = rhs_t - (i*ut)*kx_*tracer
    if (use_forcing_t)   rhs_t = rhs_t + Markovian(kf_min_t,&
            kf_max_t,forc_coef_t,forc_corr,force_ot,norm_forcing_t,tracer)
    rhs_t = filter_t*rhs_t
    
  end subroutine Get_rhs_t

  !******************* END OF PROGRAM ********************************* 

end program qg_driver
