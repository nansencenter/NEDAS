module transform_tools            !-*-f90-*-

  ! Contains procedures transforming grid<->wave and for getting
  ! Arakawa's jacobian via use of the transform.  Uses staggered
  ! grid to remove aliasing.
  !
  ! Routines:  Init_transform, Spec2grid, Grid2spec
  !
  ! Dependencies: fft_pak (one of the files of form fft_xxx.f90 where 
  !               xxx is the machine type, or 'port' for portable)
  !
  ! Algorithm by Geoff Vallis (c1991) and refitted for F90 by Shafer Smith
  ! (c1998).  Redone again in 2000 by SS.
  !

  implicit none
  private
  save

  real*8,parameter                        :: pi = 3.14159265358979
  complex,parameter                       :: i = (0.,1.)
  complex,dimension(:,:),allocatable      :: exx
  real,dimension(:,:),allocatable         :: sgn
  integer,dimension(:),allocatable        :: kxv,kxup,kxdn,kyv,kyup,kydn
  integer                                 :: nx, ny, nz, nkx, nky, kmax

  public :: Init_transform, Spec2grid_cc, Spec2grid, Grid2spec, Jacob, &
            ir_prod, ir_pwr

  interface Grid2spec
     module procedure Grid2spec_cc3, Grid2spec_cc2
     module procedure Grid2spec_rc2, Grid2spec_rc3
  end interface
  interface ir_prod
     module procedure ir_prod2, ir_prod3
  end interface
  interface ir_pwr
     module procedure ir_pwr2, ir_pwr3
  end interface
  interface jacob
     module procedure jacob2, jacob3
  end interface
  interface Spec2grid_cc
     module procedure Spec2grid_cc2, Spec2grid_cc3
  end interface

contains

  !*************************************************************************
  
  subroutine Init_transform(kmaxi,nzi)

    ! Set up index arrays for transform routines and initialize fft

    use fft_pak, only: Init_fft

    integer,intent(in)   :: kmaxi,nzi
    integer              :: kx, ky, n, m

    kmax = kmaxi; nz = nzi
    nx = 2*(kmax+1); ny = 2*(kmax+1); nkx = 2*kmax+1; nky = kmax+1
    allocate(sgn(nx,nx),exx(nkx,nky))
    allocate(kxv(nkx),kxup(nkx),kxdn(nkx),kyv(nky),kyup(nky),kydn(nky))
    kxv = (/ (kx, kx=-kmax,kmax) /)
    kyv = (/ (ky, ky=0,kmax) /)
    kxup =  kxv + kmax + 2
    kyup =  kyv + kmax + 2
    kxdn = -kxv + kmax + 2
    kydn = -kyv + kmax + 2
    exx  = cexp(cmplx(0.0,pi*(spread(kxv,2,nky)+spread(kyv,1,nkx))/nx))
    do n=1,ny
       sgn(:,n) = (/ ((-1)**(m+n), m=1,nx) /)
    enddo
    call Init_fft(kmax)

  end subroutine Init_transform

  !*************************************************************************

  function Jacob3(fk,gk) result(jack)

    ! This routine returns the jacobian of arrarys f() and g().

    use op_rules, only: operator(*)

    complex,dimension(-kmax:kmax,0:kmax,nz),intent(in)  :: fk,gk
    complex,dimension(-kmax:kmax,0:kmax,nz)             :: jack
    complex,dimension(nx,ny,nz)                         :: f,g,temp

    f = Spec2grid_cc3(i*spread(kxv,2,nky)*fk)
    g = Spec2grid_cc3(i*spread(kyv,1,nkx)*gk)
    temp = ir_prod3(f,g)
    
    f = Spec2grid_cc3(i*spread(kyv,1,nkx)*fk)
    g = Spec2grid_cc3(i*spread(kxv,2,nky)*gk)
    temp = temp - ir_prod3(f,g)
    
    jack = Grid2spec(temp)

  end function Jacob3

  !*************************************************************************

  function Jacob2(fk,gk) result(jack)

    ! This routine returns the jacobian of arrarys f() and g().

    use op_rules, only: operator(*)

    complex,dimension(-kmax:kmax,0:kmax),intent(in)  :: fk,gk
    complex,dimension(-kmax:kmax,0:kmax)             :: jack
    complex,dimension(nx,ny)                         :: f,g,temp

    f = Spec2grid_cc2(i*spread(kxv,2,nky)*fk)
    g = Spec2grid_cc2(i*spread(kyv,1,nkx)*gk)
    temp = ir_prod2(f,g)
    
    f = Spec2grid_cc2(i*spread(kyv,1,nkx)*fk)
    g = Spec2grid_cc2(i*spread(kxv,2,nky)*gk)
    temp = temp - ir_prod2(f,g)
    
    jack = Grid2spec_cc2(temp)

  end function Jacob2

  !*******************************************************************

  function Spec2grid_cc2(wf) result(physfield)

    ! Spectral to physical transform, passing straight physical field
    ! to real part of complex output variable (physfield), and same
    ! physical field *on staggered grid* to *imaginary* component
    ! of result variable.

    use fft_pak,  only: fft
    use op_rules, only: operator(*),operator(+),operator(-)

    complex,dimension(nkx,nky),intent(in)  :: wf
    complex,dimension(nkx,nky)             :: wavefield
    complex,dimension(nx,ny)               :: physfield

    wavefield = wf
    physfield = 0.
    wavefield(1:kmax+1,1)= conjg(wavefield(nkx:kmax+1:-1,1))
    physfield(kxup,kyup) = wavefield + i*(exx*wavefield)
    physfield(kxdn,kydn) = conjg(wavefield - i*(exx*wavefield))
    physfield = fft(physfield,-1)
    physfield = sgn*physfield

  end function Spec2grid_cc2

  !*******************************************************************

  function Spec2grid_cc3(wf) result(physfield)

    ! Spectral to physical transform, passing straight physical field
    ! to real part of complex output variable (physfield), and same
    ! physical field *on staggered grid* to *imaginary* component
    ! of result variable.

    use fft_pak,  only: fft
    use op_rules, only: operator(*),operator(+),operator(-)

    complex,dimension(nkx,nky,nz),intent(in)  :: wf
    complex,dimension(nkx,nky,nz)             :: wavefield
    complex,dimension(nx,ny,nz)               :: physfield

    wavefield = wf
    physfield = 0.
    wavefield(1:kmax+1,1,:)= conjg(wavefield(nkx:kmax+1:-1,1,:))
    physfield(kxup,kyup,:) = wavefield + i*(exx*wavefield)
    physfield(kxdn,kydn,:) = conjg(wavefield - i*(exx*wavefield))
    physfield = fft(physfield,-1)
    physfield = sgn*physfield

  end function Spec2grid_cc3

  !*************************************************************************

  function Spec2grid(wf) result(physfield)

    ! Spectral to physical transform, with type real result variable
    ! (just for diagnostic use since it throws away staggered grid
    ! field).

    use fft_pak,  only: fft
    use op_rules, only: operator(*),operator(+),operator(-)

    complex,dimension(nkx,nky,nz),intent(in)  :: wf
    complex,dimension(nkx,nky,nz)             :: wavefield
    complex,dimension(nx,ny,nz)               :: pfc
    real,dimension(nx,ny,nz)                  :: physfield

    pfc = 0.    
    wavefield = wf
    wavefield(1:kmax+1,1,:)= conjg(wavefield(nkx:kmax+1:-1,1,:))
    pfc(kxup,kyup,:) = wavefield + i*(exx*wavefield)
    pfc(kxdn,kydn,:) = conjg(wavefield - i*(exx*wavefield))
    pfc = fft(pfc,-1)
    physfield = real(sgn*pfc)

  end function Spec2grid

  !*************************************************************************

  function Grid2spec_cc2(pf) result(wavefield)

    ! Physical to spectral transform:  assumes complex input variable 
    ! contains field as real part and same field on staggered grid
    ! in imaginary part.  

    use fft_pak,  only: fft
    use op_rules, only: operator(*),operator(+),operator(-)

    complex,dimension(nx,ny),intent(in)     :: pf
    complex,dimension(nx,ny)                :: physfield
    complex,dimension(nkx,nky)              :: wavefield

    wavefield = 0.    
    physfield = sgn*pf
    physfield = fft(physfield,1)
    wavefield = real(physfield(kxup,kyup)+physfield(kxdn,kydn)) &
              +  i*aimag(physfield(kxup,kyup)-physfield(kxdn,kydn)) &
              + (aimag(physfield(kxup,kyup)+physfield(kxdn,kydn)) &
              +  i*real(-physfield(kxup,kyup)+physfield(kxdn,kydn))) &
                 *conjg(exx)
    wavefield = .25*wavefield

  end function Grid2spec_cc2

  !*************************************************************************

  function Grid2spec_cc3(pf) result(wavefield)

    ! Physical to spectral transform:  assumes complex input variable 
    ! contains field as real part and same field on staggered grid
    ! in imaginary part.  

    use fft_pak,  only: fft
    use op_rules, only: operator(*),operator(+),operator(-)

    complex,dimension(nx,ny,nz),intent(in)     :: pf
    complex,dimension(nx,ny,nz)                :: physfield
    complex,dimension(nkx,nky,nz)              :: wavefield

    wavefield = 0.    
    physfield = sgn*pf
    physfield = fft(physfield,1)
    wavefield = real(physfield(kxup,kyup,:)+physfield(kxdn,kydn,:)) &
              +  i*aimag(physfield(kxup,kyup,:)-physfield(kxdn,kydn,:)) &
              + (aimag(physfield(kxup,kyup,:)+physfield(kxdn,kydn,:)) &
              +  i*real(-physfield(kxup,kyup,:)+physfield(kxdn,kydn,:))) &
                 *conjg(exx)
    wavefield = .25*wavefield

  end function Grid2spec_cc3

  !*************************************************************************

  function Grid2spec_rc2(pf) result(wavefield)

    ! Physical to spectral transform - just for diagnostic use, as 
    ! product is not dealiased.

    use fft_pak,  only: fft
    use op_rules, only: operator(*),operator(+),operator(-)

    real,dimension(nx,ny),intent(in)     :: pf
    complex,dimension(nx,ny)             :: physfield
    complex,dimension(nkx,nky)           :: wavefield

    wavefield = 0.    
    physfield = sgn*cmplx(pf,pf)
    physfield = fft(physfield,1)
    wavefield = real(physfield(kxup,kyup)+physfield(kxdn,kydn)) &
              +  i*aimag(physfield(kxup,kyup)-physfield(kxdn,kydn)) &
              + (aimag(physfield(kxup,kyup)+physfield(kxdn,kydn)) &
              +  i*real(-physfield(kxup,kyup)+physfield(kxdn,kydn))) &
                 *conjg(exx)
    wavefield = .25*wavefield

  end function Grid2spec_rc2

  !*************************************************************************

  function Grid2spec_rc3(pf) result(wavefield)

    ! Physical to spectral transform - just for diagnostic use, as 
    ! product is not dealiased.

    use fft_pak,  only: fft
    use op_rules, only: operator(*),operator(+),operator(-)

    real,dimension(nx,ny,nz),intent(in)     :: pf
    complex,dimension(nx,ny,nz)             :: physfield
    complex,dimension(nkx,nky,nz)           :: wavefield

    wavefield = 0.    
    physfield = sgn*cmplx(pf,pf)
    physfield = fft(physfield,1)
    wavefield = real(physfield(kxup,kyup,:)+physfield(kxdn,kydn,:)) &
              +  i*aimag(physfield(kxup,kyup,:)-physfield(kxdn,kydn,:)) &
              + (aimag(physfield(kxup,kyup,:)+physfield(kxdn,kydn,:)) &
              +  i*real(-physfield(kxup,kyup,:)+physfield(kxdn,kydn,:))) &
                 *conjg(exx)
    wavefield = .25*wavefield

  end function Grid2spec_rc3

  !*******************************************************************

  function ir_prod2(f,g) result(prod)

    ! This is for doing the special multiply of physical fields,
    ! assuming that fields on the staggered grid are stored as the
    ! imaginary component.

    complex,dimension(:,:),intent(in)       :: f,g
    complex,dimension(size(f,1),size(f,2))  :: prod
    
    prod = cmplx(real(f)*real(g),aimag(f)*aimag(g))

  end function ir_prod2

  !*******************************************************************

  function ir_prod3(f,g) result(prod)

    ! This is for doing the special multiply of physical fields,
    ! assuming that fields on the staggered grid are stored as the
    ! imaginary component.

    complex,dimension(:,:,:),intent(in)               :: f,g
    complex,dimension(size(f,1),size(f,2),size(f,3))  :: prod
    
    prod = cmplx(real(f)*real(g),aimag(f)*aimag(g))

  end function ir_prod3

  !*******************************************************************

  function ir_pwr2(f,pwr) result(prod)

    ! This is for doing the special power raise of physical fields,
    ! assuming that fields on the staggered grid are stored as the
    ! imaginary component.

    complex,dimension(:,:),intent(in)       :: f
    real,intent(in)                         :: pwr
    complex,dimension(size(f,1),size(f,2))  :: prod
    
    ! If this isn't the stupidest damn thing you've seen,
    ! well i'll just be damned.  This is another fucking bug
    ! in the compiler - death to SGI!

    if (pwr==2.) then
       prod = cmplx(real(f)**2.,aimag(f)**2.)
    elseif (pwr==.5) then
       prod = cmplx(real(f)**.5,aimag(f)**.5)
    endif
       
  end function ir_pwr2

  !*******************************************************************

  function ir_pwr3(f,pwr) result(prod)

    ! This is for doing the special power raise of physical fields,
    ! assuming that fields on the staggered grid are stored as the
    ! imaginary component.

    complex,dimension(:,:,:),intent(in)               :: f
    real,intent(in)                                   :: pwr
    complex,dimension(size(f,1),size(f,2),size(f,3))  :: prod
    
    if (pwr==2.) then
       prod = cmplx(real(f)**2.,aimag(f)**2.)
    elseif (pwr==.5) then
       prod = cmplx(real(f)**.5,aimag(f)**.5)
    endif

  end function ir_pwr3

  !*************************************************************************

end module transform_tools



