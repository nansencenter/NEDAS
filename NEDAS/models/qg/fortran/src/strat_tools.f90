module strat_tools        

  !************************************************************************
  ! Contains all the tools for setting up the stratification.
  !
  ! Routines: Make_stc_strat, Make_exp_strat, Get_z, 
  !           Get_vmodes, Get_tripint, Layer2mode, Mode2layer
  !
  ! Dependencies: io_tools, eig_pak, numerics_lib
  !************************************************************************

  implicit none
  save

contains

  !*************************************************************************

   function Strat_params(dz,drho,F,Fe,surface_bc,nv) result(op)

    !************************************************************************
    ! Set coefficients for tridiagonal stratification operator 'op',
    ! where q = del^2 psi + op psi
    ! op(:,-1) = subdiagonal, op(:,0) = diagonal, and op(:,1) = superdiagonal
    !************************************************************************
     
    use io_tools, only: Message

    real,dimension(:),intent(in)                   :: dz
    real,dimension(1-(nv-size(dz)):size(dz)-1),intent(in)  :: drho
    real,intent(in)                                :: F, Fe
    character(*),intent(in)                        :: surface_bc
    integer,intent(in)                             :: nv
    real,dimension(1-(nv-size(dz)):size(dz),-1:1)  :: op
    integer                                        :: nz

    nz = size(dz)

    if (any(dz==0))   call Message('Error:Strat_params: some dz=0',fatal='y')
    if (any(drho==0)) call Message('Error:Strat_params: some drho=0',fatal='y')
    
    op = 0.
    op(2:nz,-1)  = F/(dz(2:nz)*drho(1:nz-1))
    op(1:nz-1,1) = F/(dz(1:nz-1)*drho(1:nz-1))
    
    ! Default is rigid_lid
    select case (trim(surface_bc))
    case('periodic')
       ! Set off-matrix values to wrap-around - see tridiag_cyc in numerics_lib
       op(1,-1) = op(1,1)
       op(nz,1) = op(nz,-1)
    case ('surf_buoy')
       ! Assume psi at surface is in a delta sheet there, so distance is
       ! just half the top layer thickness, dz(1)/2.  Linearly extrapolate
       ! drho to get assumed delta rho from middle of top layer (where 
       ! psi(1) is calculated) to very top.
       op(1,-1) = F/(dz(1)*drho(0))
       op(0,0) = 1./dz(1)
       op(0,1) = -1./dz(1)
    end select
    op(1:nz,0) = -op(1:nz,-1)-op(1:nz,1)
!    op(1,0) = op(1,0) - Fe/dz(1)   ! Free upper surface; Fe = F*drho(1)/drho(0)
    op(nz,0) = op(nz,0) - Fe/dz(nz) ! Free lower surface; Fe = F*drho(nz)/drho(nz+1)

  end function Strat_params

  !*************************************************************************

  subroutine Make_strat(dz,rho,deltc,strat_type,alphai)

    !************************************************************************
    ! Produces stratification profile and vertical discretization. deltc = 1/e 
    !     value of curvature of rho with depth (deltc ~ 
    !     [depth of TC]/[depth of fluid]) Returns layer thicknesses, dz, 
    !     and layer center positions, zl, and the densities, rho.
    !     Layer interfaces are set to lie at exactly the zero-crossings of 
    !     the highest vertical mode available for the given vertical reso-
    !     lution, i.e. nlevs-1.  This is as per results of A. Beckman, JPO 18,
    !     pp. 1354--1371.   For surface intensified
    !     stratification, we first find the modes for a system with the shape
    !     of rho determined by deltc, alpha and rhotop and with some vertical 
    !     resolution sufficiently higher than any that would be used in the
    !     full model, then use a zero-finding
    !     algorithm to get the vertical positions of 
    !     the crossings, and finally convert this back to dz and get the
    !     rho profile for this discretization.
    !************************************************************************

    use numerics_lib, only: findzeros
    
    real,dimension(:),intent(OUT)  :: dz,rho
    real,intent(IN)                :: deltc
    character(*),intent(in)        :: strat_type
    real,intent(IN),optional       :: alphai

    ! Local
    integer,parameter              :: nzh   = 100
    real                           :: alpha = 0.0005
    real,dimension(nzh)            :: dzh,zh,rhoh,kzh,zzero
    real,dimension(nzh-1)          :: drhoh
    real,dimension(nzh,nzh)        :: vmodeh
    real,dimension(size(dz))       :: zl
    real,dimension(size(dz)-1)     :: zi
    integer                        :: nz

    if (present(alphai)) alpha = alphai   ! To make gentle slope at depth
    nz = size(dz)

    ! Make high vertical resolution strat functions
    dzh = 1./float(nzh)                ! High res layer thicknesses - lin
    zh = Get_z(dzh)                    ! High res layer center pos'ns
    select case (trim(strat_type))
    case ('stc');  rhoh = (1+(tanh(zh/deltc))**2)*(1-alpha*zh)
    case ('exp');  rhoh = (1-alpha*zh) +(1-exp(zh/deltc))
    end select
    drhoh = rhoh(2:nzh)-rhoh(1:nzh-1)

    call Get_vmodes(dzh,drhoh,1.,0.,kzh,vmodeh)   ! High res strat modes
    
    zzero = Findzeros(vmodeh(:,nz),zh)          ! Interface positions
    zi = zzero(1:nz-1)

    dz(1) = abs(zi(1))                                    
    dz(2:nz-1) = abs(zi(2:nz-1)-zi(1:nz-2))    ! Layer thicknesses
    dz(nz) = abs(-1-zi(nz-1))                  ! z=-1 at bottom of ocean
    zl = Get_z(dz)                             ! Positions of layer centers

    ! Density subsampled on new z 
    select case (trim(strat_type))
    case ('stc');  rho = (1+(tanh(zl/deltc))**2)*(1-alpha*zl)
    case ('exp');  rho = (1-alpha*zl) +(1-exp(zl/deltc))
    end select
    
  end subroutine Make_strat

  !*************************************************************************

  function Get_z(dz)

    !************************************************************************
    ! Convert the layer thicknesses into the vertical positions of the 
    ! centers of each layer, i.e. the proper vertial coordinate on which
    ! to calculate the fields (the surface is at z = 0).
    !************************************************************************

    real,dimension(:),intent(in)      :: dz
    real,dimension(size(dz))          :: Get_z
    integer                           :: n

    Get_z(1) = dz(1)/2.
    do n=2,size(dz)
       Get_z(n) = sum(dz(1:n-1))+dz(n)/2.
    enddo
    Get_z = -Get_z

  end function Get_z

  !*************************************************************************

  subroutine Get_vmodes(dz,drho,F,Fe,kz,vmode,surface_bc)

    !************************************************************************
    ! Calculate the vertical (stratification) modes for a given density
    ! structure and vertical discretization.  i.e. solve the Sturm-Liouville
    ! eigenvalue problem for the discretized stratification operator,
    ! d/dz ( B psi/dz) == op psi = -kz^2 psi, where B is a place holder for
    ! the nondimensional, present version of f^2/N^2, and op can also be
    ! viewed as the vorticity stretching operator in the q - psi relation,
    ! q = del^2 psi + op psi
    !************************************************************************

    use io_tools,     only: Message
    use eig_pak,      only: eig
    use numerics_lib, only: sortvec, tri2mat

    real,dimension(:),intent(in)       :: dz, drho
    real,intent(in)                    :: F,Fe
    real,dimension(:),intent(out)      :: kz
    real,dimension(:,:),intent(out)    :: vmode
    character(*),intent(in),optional   :: surface_bc

    ! Local

    real,dimension(:,:),allocatable    :: vmodeout,op,opv
    integer,dimension(:),allocatable   :: ind_map
    integer                            :: n,nz,ierr
    real                               :: alpha, a, b

    nz = size(dz)

    if (nz>2) then

       allocate(op(nz,nz),opv(nz,-1:1),vmodeout(nz,nz),ind_map(nz))

       select case (trim(surface_bc))
       case ('periodic');     opv = strat_params(dz,drho,F,Fe,surface_bc,nz)
       case default;          opv = strat_params(dz,drho,F,Fe,'rigid_lid',nz)
       end select
       
       op = tri2mat(opv)

       ! Get the evecs and evals of op

       call eig(op, eval=kz, evec=vmode, info=ierr) 
       if (ierr/=0) call Message('Get_vmodes: eigensolve failure: ', &
            tag=ierr, fatal='y')

       kz = -kz
       where (kz > 0.) 
          kz = sqrt(kz)
       elsewhere
          kz = 0.
       endwhere

       call Sortvec(kz,ind_map)     ! Sort kz and return index map for vmodes
!       kz(1) = 0.                   ! Make sure BT kz is 0 (not just tiny)

       vmodeout = vmode(:,ind_map)  ! Sort vmodes along sorted kz
       vmode = vmodeout

       ! Now do normalization with weighting from layer thicknesses,
       ! since <vmode(:,i),vmode(:,j)>=(vmode(:,i).*dz)'*vmode(:,j)=delta(i,j)
       
       do n = 1,nz
          vmode(:,n) = vmode(:,n)/sqrt(dot_product(vmode(:,n)*dz,vmode(:,n)))
          if (vmode(1,n)<0) vmode(:,n) = -vmode(:,n)
       enddo
       
       deallocate(op,opv,vmodeout,ind_map) 

    elseif (nz==2) then             ! Use analytic result for 2 layer case

       if ((Fe/=0).and.(dz(1)/=dz(2))) call Message('Error:Get_vmodes:2 layer&
             & free surface assumes equal layer thicknesses for modes',fatal='y')
       if (Fe/=0) then
          alpha = Fe/F
          a = sqrt(1+(alpha/2)**2)
          b = alpha/2
          kz = (/ sqrt((F/dz(1))*(1 + b - a)), sqrt((F/dz(1))*(1 + b + a)) /)
          vmode(:,1) = (/  1. , a+b /)
          vmode(:,2) = (/ -1. , a-b /)
       else
          kz = (/ 0. , sqrt(F)/sqrt(dz(1)*(1-dz(1))) /)
          vmode(:,1) = (/ 1. , 1. /)
          vmode(:,2) = (/ sqrt((1-dz(1))/dz(1)) , -sqrt(dz(1)/(1-dz(1))) /)
       endif

    endif

  end subroutine Get_vmodes

  !*********************************************************************

  function Get_tripint(dz,rho,surface_bc,hfi,drt,drb) result(tripint)

    !************************************************************************
    ! Calculate the tripple interaction coefficient for the vertical
    ! modes with layers of thicknesses dz and density rho.  
    ! MODIFIED to use higher resolution (interpolated) density profile
    ! to get de-aliased triple interaction coefficient.
    !************************************************************************

    use io_tools,     only: Message, Write_field
    use numerics_lib, only: spline, splint

    real,dimension(:),intent(in)               :: dz,rho
    character(*),intent(in)                    :: surface_bc
    integer,intent(in),optional                :: hfi
    real,intent(in),optional                   :: drt,drb
    real,dimension(size(dz),size(dz),size(dz)) :: tripint

    ! Local variables

    integer                         :: hf = 10
    integer                         :: i,j,k, nz, nzh, n
    real,dimension(:),allocatable   :: dzh,zh,z,rhoh,kzh,d2rho,drhoh
    real,dimension(:,:),allocatable :: vmodeh

    if (present(hfi)) hf = hfi
    nz = size(dz)

    if (nz>2) then

       nzh = hf*nz
       allocate(vmodeh(nzh,nzh),dzh(nzh),zh(nzh),z(nz),rhoh(nzh),kzh(nzh))
       vmodeh=0.; dzh=0.; zh=0.; z=0; rhoh=0.; kzh=0.
       allocate(d2rho(nz),drhoh(nzh-1))
       d2rho=0; drhoh=0.

       z = Get_z(dz)
       dzh = 1./float(nzh)     ! Make higher res but uniform grid
       if (dzh(1) > minval(dz)) then
          call Write_field(dz,'dz')
          call Write_field(dzh,'dzh')
          call Message('Get_tripint: grid too coarse',fatal='y')
       endif
       zh = Get_z(dzh)
       if (drt>-1000) then
          d2rho = spline(z,rho,drt,drb)
       else
          d2rho = spline(z,rho)
       endif

       do n = 1,nzh            ! Interpolate to higher res density
          rhoh(n) = Splint(-z,rho,d2rho,-zh(n))
       enddo
       drhoh = rhoh(2:nzh)-rhoh(1:nzh-1)
       if (any(drhoh<=0.)) then
          call Write_field(rho,'rho')
          call Write_field(dz,'dz')
          call Write_field(rhoh,'rhoh')
          call Write_field(d2rho,'d2rho')
          call Write_field(dzh,'dzh')
          call Message('Error: Some drhoh<0 in get_tripint',fatal='y')
       endif
       
       call Get_vmodes(dzh,drhoh,1.,0.,kzh,vmodeh,surface_bc) 
       
       do k = 1,nz             ! Use only nz modes
          do j = 1,nz
             do i = 1,nz
                tripint(i,j,k) = sum(dzh*vmodeh(:,i)*vmodeh(:,j)*vmodeh(:,k));
             enddo
          enddo
       enddo
       
       deallocate(vmodeh,zh,dzh,z,rhoh,kzh,d2rho,drhoh)

    elseif (nz==2) then

       tripint(1,1,1) = 1.; tripint(1,2,2) = 1.; tripint(2,1,2) = 1.;
       tripint(2,2,1) = 1.
       tripint(2,2,2) = (1-2*dz(1))/sqrt(dz(1)*(1-dz(1)))

    endif

  end function Get_tripint

  !*********************************************************************

  function Layer2mode(f,vmode,dz) result(fm)

    !************************************************************************
    ! Project layered field onto vertical modes - third index
    ! will now denote mode: 1 = BT, 2 = BC1, etc...
    !************************************************************************

    complex,dimension(:,:,:),intent(in)              :: f
    complex,dimension(size(f,1),size(f,2),size(f,3)) :: fm
    real,dimension(size(f,3),size(f,3)),intent(in)   :: vmode
    real,dimension(size(f,3)),intent(in)             :: dz
    integer                                          :: ix,iy,n,nx,ny,nz

    nx = size(f,1); ny = size(f,2); nz = size(f,3)

    do n = 1,nz
       do iy = 1,ny
          do ix = 1,nx
             fm(ix,iy,n) = dot_product((vmode(:,n)*dz),f(ix,iy,:))
          enddo
       enddo
    enddo

  end function Layer2mode

  !*********************************************************************

  function Mode2layer(fm,vmode) result(f)

    !************************************************************************
    ! Project modal field onto layers - third index will now
    ! denote layer
    !************************************************************************

    complex,dimension(:,:,:),intent(in)                 :: fm
    complex,dimension(size(fm,1),size(fm,2),size(fm,3)) :: f
    real,dimension(size(fm,3),size(fm,3)),intent(in)    :: vmode
    integer                                             :: ix,iy,nx,ny

    nx = size(f,1); ny = size(f,2)

    do iy = 1,ny             ! Project modes onto layers
       do ix = 1,nx
          f(ix,iy,:) = matmul(vmode,fm(ix,iy,:))
       enddo
    enddo

  end function Mode2layer

  !*********************************************************************

end module strat_tools
