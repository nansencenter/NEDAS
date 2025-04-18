module fft_pak                    
! Arrow of time adjusted......8 Nov 04
!
! This one is for the FFTW 3.0.1 fma package, uniprocessor, double precision
! This one tries to use the same plan_f, plan_b all the time. 
!
! Routine: Getfft
!

!  Oliver's notes: Here it is.  There is one explicit kind selection
! for the Fortran integer plan that is used as a C pointer by fftw; this
! must be 32 bit on my G4.  Otherwise this is almost exactly as fftw2,
! so I don't know why they made such a fuss about not making it
! backwards compatible.


! This is from fftw3.f:
      INTEGER FFTW_R2HC
      PARAMETER (FFTW_R2HC=0)
      INTEGER FFTW_HC2R
      PARAMETER (FFTW_HC2R=1)
      INTEGER FFTW_DHT
      PARAMETER (FFTW_DHT=2)
      INTEGER FFTW_REDFT00
      PARAMETER (FFTW_REDFT00=3)
      INTEGER FFTW_REDFT01
      PARAMETER (FFTW_REDFT01=4)
      INTEGER FFTW_REDFT10
      PARAMETER (FFTW_REDFT10=5)
      INTEGER FFTW_REDFT11
      PARAMETER (FFTW_REDFT11=6)
      INTEGER FFTW_RODFT00
      PARAMETER (FFTW_RODFT00=7)
      INTEGER FFTW_RODFT01
      PARAMETER (FFTW_RODFT01=8)
      INTEGER FFTW_RODFT10
      PARAMETER (FFTW_RODFT10=9)
      INTEGER FFTW_RODFT11
      PARAMETER (FFTW_RODFT11=10)
      INTEGER FFTW_FORWARD
      PARAMETER (FFTW_FORWARD=-1)
      INTEGER FFTW_BACKWARD
      PARAMETER (FFTW_BACKWARD=+1)
      INTEGER FFTW_MEASURE
      PARAMETER (FFTW_MEASURE=0)
      INTEGER FFTW_DESTROY_INPUT
      PARAMETER (FFTW_DESTROY_INPUT=1)
      INTEGER FFTW_UNALIGNED
      PARAMETER (FFTW_UNALIGNED=2)
      INTEGER FFTW_CONSERVE_MEMORY
      PARAMETER (FFTW_CONSERVE_MEMORY=4)
      INTEGER FFTW_EXHAUSTIVE
      PARAMETER (FFTW_EXHAUSTIVE=8)
      INTEGER FFTW_PRESERVE_INPUT
      PARAMETER (FFTW_PRESERVE_INPUT=16)
      INTEGER FFTW_PATIENT
      PARAMETER (FFTW_PATIENT=32)
      INTEGER FFTW_ESTIMATE
      PARAMETER (FFTW_ESTIMATE=64)
      INTEGER FFTW_ESTIMATE_PATIENT
      PARAMETER (FFTW_ESTIMATE_PATIENT=128)
      INTEGER FFTW_BELIEVE_PCOST
      PARAMETER (FFTW_BELIEVE_PCOST=256)
      INTEGER FFTW_DFT_R2HC_ICKY
      PARAMETER (FFTW_DFT_R2HC_ICKY=512)
      INTEGER FFTW_NONTHREADED_ICKY
      PARAMETER (FFTW_NONTHREADED_ICKY=1024)
      INTEGER FFTW_NO_BUFFERING
      PARAMETER (FFTW_NO_BUFFERING=2048)
      INTEGER FFTW_NO_INDIRECT_OP
      PARAMETER (FFTW_NO_INDIRECT_OP=4096)
      INTEGER FFTW_ALLOW_LARGE_GENERIC
      PARAMETER (FFTW_ALLOW_LARGE_GENERIC=8192)
      INTEGER FFTW_NO_RANK_SPLITS
      PARAMETER (FFTW_NO_RANK_SPLITS=16384)
      INTEGER FFTW_NO_VRANK_SPLITS
      PARAMETER (FFTW_NO_VRANK_SPLITS=32768)
      INTEGER FFTW_NO_VRECURSE
      PARAMETER (FFTW_NO_VRECURSE=65536)
      INTEGER FFTW_NO_SIMD
      PARAMETER (FFTW_NO_SIMD=131072)

integer*8                :: plan_f,plan_b
!integer (kind=3)                :: plan_f,plan_b
integer                  :: hres
integer,dimension(2)     :: nsize

private
save

public :: Init_fft, fft

  interface fft
     module procedure fft2, fft3
  end interface

contains

  !********************************************************************

  subroutine Init_fft(kmax)

    ! Initialize fft routine
    ! Does little for version 3
    integer,intent(in)                     :: kmax
    complex,dimension(:,:), allocatable    :: f,fr

    hres = 2*kmax+2
    nsize(1) = hres
    nsize(2) = hres
    allocate(f(nsize(1),nsize(2)))
    allocate(fr(nsize(1),nsize(2)))

    call dfftw_plan_dft_2d(plan_b,nsize(1),nsize(2),f,fr,&
          FFTW_BACKWARD,FFTW_PATIENT) 
    call dfftw_plan_dft_2d(plan_f,nsize(1),nsize(2),f,fr,&
          FFTW_FORWARD,FFTW_PATIENT)
    deallocate(f)
    deallocate(fr)

  end subroutine Init_fft

  !********************************************************************

  function fft2(f,dirn) result(fr)

    ! Calculate 2d complex to complex fft.  dirn = +1 or -1.
    ! these values correspond to sign of exponent in spectral
    ! sum - arrow of time?

  complex,dimension(:,:)                  :: f
  complex,dimension(size(f,1),size(f,2))  :: fr
  integer,intent(in)                      :: dirn
  real                                    :: scale=1.0

  if (dirn==-1) then
     scale=1.0
     call dfftw_execute_dft(plan_b,f,fr)
  elseif (dirn==1)  then
     scale=1.0/float(hres)**2
     call dfftw_execute_dft(plan_f,f,fr)
  endif

  fr = scale*fr

  end function fft2

  !********************************************************************

  function fft3(f,dirn) result(fr)

    ! Calculate 3d complex to complex fft.  dirn = +1 or -1.
    ! these values correspond to sign of exponent in spectral
    ! sum (see man page on ccfft2d).

    complex,dimension(:,:,:)                          :: f
    complex,dimension(size(f,1),size(f,2),size(f,3))  :: fr
    integer,intent(in)                                :: dirn
    real                                              :: scale=1.0
    do n=1,size(f,3)
      if (dirn==-1) then
         scale=1.0
         call dfftw_execute_dft(plan_b,f(:,:,n),fr(:,:,n))
      elseif (dirn==1)  then
         scale=1.0/float(hres)**2
         call dfftw_execute_dft(plan_f,f(:,:,n),fr(:,:,n))
      endif
    enddo

    fr = scale*fr

  end function fft3

  !********************************************************************

end module fft_pak
