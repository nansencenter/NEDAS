module op_rules         

  ! This module adds functionality to arithmetic operators so 
  ! that arrays of different rank can be manipulated without using loops
  !
  ! In general, use of the operators +,-,* assumes the following for A <op> B
  ! or B <op> A:
  ! 
  ! 1.  A = A(:)   and B = B(:,:,:) => A is conformable with B(i,j,:)
  !
  ! 2.  A = A(:,:) and B = B(:,:,:) => A is conformable with B(:,:,i)
  !
  ! Also, squeeze operator removes 3rd dimension of any rank 3 array.
  !
  ! >>> Need to add rest of functionality for +, -, and all of it for /.
  !     Also need to add error checking to make sure dimensions are
  !     conformable.

  implicit none 
  private

  public :: squeeze, operator(+), operator(-), operator(*)

  interface squeeze
     module procedure squeezec, squeezer
  end interface
  interface operator(*)
     module procedure array1r_x_array2r, array2r_x_array1r
     module procedure array1r_x_array2c, array2c_x_array1r
     module procedure array1r_x_array3r, array3r_x_array1r  
     module procedure array1r_x_array3c, array3c_x_array1r
     module procedure array2r_x_array3r, array3r_x_array2r
     module procedure array2r_x_array3c, array3c_x_array2r
     module procedure array2c_x_array3r, array3r_x_array2c
     module procedure array2c_x_array3c, array3c_x_array2c
  end interface
  interface operator(-)
     module procedure array1r_m_array2r, array2r_m_array1r
     module procedure array2c_m_array3c, array3c_m_array2c
  end interface
  interface operator(+)
     module procedure array1r_p_array2r, array2r_p_array1r 
     module procedure array2c_p_array3c, array3c_p_array2c
  end interface

contains

  !***************************** SQUEEZE *********************************

  function squeezec(array3c)
    ! Assumes 3rd dim of array3c is a singleton dimension

    use io_tools, only: Message

    complex,dimension(:,:,:),intent(in)                 :: array3c
    complex,dimension(size(array3c,1),size(array3c,2))  :: squeezec

    if (size(array3c,3)/=1) then
       call Message('Call to squeezec is removing non-singleton dim',fatal='y')
    endif
    squeezec(:,:) = array3c(:,:,1)

  end function squeezec

  !*************

  function squeezer(array3r)
    ! Assumes 3rd dim of array3r is a singleton dimension

    use io_tools, only: Message

    real,dimension(:,:,:),intent(in)                     :: array3r
    real,dimension(size(array3r,1),size(array3r,2))      :: squeezer

    if (size(array3r,3)/=1) then
       call Message('Call to squeezer is removing non-singleton dim',fatal='y')
    endif
    squeezer(:,:) = array3r(:,:,1)

  end function squeezer

  !****************************** MULTIPLY *********************************

  function array1r_x_array3c(array1r,array3c)
    ! Assumes array1r is some fixed real vector in the z (dim 3) direction
    ! by which we want to multiply each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    real,dimension(:),intent(in)        :: array1r
    complex,dimension(:,:,:),intent(in) :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array1r_x_array3c
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array1r_x_array3c(:,:,n) = array1r(n)*array3c(:,:,n)
    enddo
  end function array1r_x_array3c

  !*************

  function array3c_x_array1r(array3c,array1r)
    ! Assumes array1r is some fixed real vector in the z (dim 3) direction
    ! by which we want to multiply each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    real,dimension(:),intent(in)        :: array1r
    complex,dimension(:,:,:),intent(in) :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array3c_x_array1r
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array3c_x_array1r(:,:,n) = array1r(n)*array3c(:,:,n)
    enddo
  end function array3c_x_array1r

  !*************

  function array1r_x_array3r(array1r,array3r)
    ! Assumes array1r is some fixed real vector in the z (dim 3) direction
    ! by which we want to multiply each level (dim 3) of real 3d array 

    use io_tools, only: Message

    real,dimension(:),intent(in)        :: array1r
    real,dimension(:,:,:),intent(in)    :: array3r
    real,dimension(size(array3r,1),size(array3r,2),size(array3r,3)) :: &
         array1r_x_array3r
    integer  :: n, nz

    nz = size(array3r,3)
    do n = 1,nz
       array1r_x_array3r(:,:,n) = array1r(n)*array3r(:,:,n)
    enddo
  end function array1r_x_array3r

  !*************

  function array3r_x_array1r(array3r,array1r)
    ! Assumes array1r is some fixed real vector in the z (dim 3) direction
    ! by which we want to multiply each level (dim 3) of real 3d array 

    use io_tools, only: Message

    real,dimension(:),intent(in)        :: array1r
    real,dimension(:,:,:),intent(in)    :: array3r
    real,dimension(size(array3r,1),size(array3r,2),size(array3r,3)) :: &
         array3r_x_array1r
    integer  :: n, nz

    nz = size(array3r,3)
    do n = 1,nz
       array3r_x_array1r(:,:,n) = array1r(n)*array3r(:,:,n)
    enddo
  end function array3r_x_array1r

  !*************

  function array2r_x_array3c(array2r,array3c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to multiply each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    real,dimension(:,:),intent(in)        :: array2r
    complex,dimension(:,:,:),intent(in)   :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array2r_x_array3c
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array2r_x_array3c(:,:,n) = array2r*array3c(:,:,n)
    enddo
  end function array2r_x_array3c

  !*************

  function array3c_x_array2r(array3c,array2r)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to multiply each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    real,dimension(:,:),intent(in)        :: array2r
    complex,dimension(:,:,:),intent(in)   :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array3c_x_array2r
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array3c_x_array2r(:,:,n) = array2r*array3c(:,:,n)
    enddo
  end function array3c_x_array2r

  !*************

  function array2r_x_array3r(array2r,array3r)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to multiply each level (dim 3) of real 3d array 

    use io_tools, only: Message

    real,dimension(:,:),intent(in)     :: array2r
    real,dimension(:,:,:),intent(in)   :: array3r
    real,dimension(size(array3r,1),size(array3r,2),size(array3r,3)) :: &
         array2r_x_array3r
    integer  :: n, nz

    nz = size(array3r,3)
    do n = 1,nz
       array2r_x_array3r(:,:,n) = array2r*array3r(:,:,n)
    enddo
  end function array2r_x_array3r

  !*************

  function array3r_x_array2r(array3r,array2r)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to multiply each level (dim 3) of real 3d array 

    use io_tools, only: Message

    real,dimension(:,:),intent(in)     :: array2r
    real,dimension(:,:,:),intent(in)   :: array3r
    real,dimension(size(array3r,1),size(array3r,2),size(array3r,3)) :: &
         array3r_x_array2r
    integer  :: n, nz

    nz = size(array3r,3)
    do n = 1,nz
       array3r_x_array2r(:,:,n) = array2r*array3r(:,:,n)
    enddo
  end function array3r_x_array2r

  !*************

  function array2c_x_array3c(array2c,array3c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to multiply each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    complex,dimension(:,:),intent(in)     :: array2c
    complex,dimension(:,:,:),intent(in)   :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array2c_x_array3c
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array2c_x_array3c(:,:,n) = array2c*array3c(:,:,n)
    enddo
  end function array2c_x_array3c

  !*************

  function array3c_x_array2c(array3c,array2c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to multiply each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    complex,dimension(:,:),intent(in)     :: array2c
    complex,dimension(:,:,:),intent(in)   :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array3c_x_array2c
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array3c_x_array2c(:,:,n) = array2c*array3c(:,:,n)
    enddo
  end function array3c_x_array2c

  !*************

  function array2c_x_array3r(array2c,array3r)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to multiply each level (dim 3) of real 3d array 

    use io_tools, only: Message

    complex,dimension(:,:),intent(in)     :: array2c
    real,dimension(:,:,:),intent(in)      :: array3r
    complex,dimension(size(array3r,1),size(array3r,2),size(array3r,3)) :: &
         array2c_x_array3r
    integer  :: n, nz

    nz = size(array3r,3)
    do n = 1,nz
       array2c_x_array3r(:,:,n) = array2c*array3r(:,:,n)
    enddo
  end function array2c_x_array3r

  !*************

  function array3r_x_array2c(array3r,array2c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to multiply each level (dim 3) of real 3d array 

    use io_tools, only: Message

    complex,dimension(:,:),intent(in)     :: array2c
    real,dimension(:,:,:),intent(in)      :: array3r
    complex,dimension(size(array3r,1),size(array3r,2),size(array3r,3)) :: &
         array3r_x_array2c
    integer  :: n, nz

    nz = size(array3r,3)
    do n = 1,nz
       array3r_x_array2c(:,:,n) = array2c*array3r(:,:,n)
    enddo
  end function array3r_x_array2c

  !*************

  function array1r_x_array2r(array1r,array2r)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! and array1r is some fixed real vector in the z (dim 3) direction

    use io_tools, only: Message

    real,dimension(:),intent(in)          :: array1r
    real,dimension(:,:),intent(in)        :: array2r
    real,dimension(size(array2r,1),size(array2r,2),size(array1r)) :: &
         array1r_x_array2r
    integer  :: n, nz

    nz = size(array1r)

    do n = 1,nz
       array1r_x_array2r(:,:,n) = array1r(n)*array2r
    enddo
  end function array1r_x_array2r

  !*************

  function array2r_x_array1r(array2r,array1r)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! and array1r is some fixed real vector in the z (dim 3) direction

    use io_tools, only: Message

    real,dimension(:),intent(in)          :: array1r
    real,dimension(:,:),intent(in)        :: array2r
    real,dimension(size(array2r,1),size(array2r,2),size(array1r)) :: &
         array2r_x_array1r
    integer  :: n, nz

    nz = size(array1r)

    do n = 1,nz
       array2r_x_array1r(:,:,n) = array1r(n)*array2r
    enddo
  end function array2r_x_array1r

  !*************

  function array1r_x_array2c(array1r,array2c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! and array1r is some fixed real vector in the z (dim 3) direction

    use io_tools, only: Message

    real,dimension(:),intent(in)             :: array1r
    complex,dimension(:,:),intent(in)        :: array2c
    complex,dimension(size(array2c,1),size(array2c,2),size(array1r)) :: &
         array1r_x_array2c
    integer  :: n, nz

    nz = size(array1r)

    do n = 1,nz
       array1r_x_array2c(:,:,n) = array1r(n)*array2c
    enddo
  end function array1r_x_array2c

  !*************

  function array2c_x_array1r(array2c,array1r)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! and array1r is some fixed real vector in the z (dim 3) direction

    use io_tools, only: Message

    real,dimension(:),intent(in)              :: array1r
    complex,dimension(:,:),intent(in)         :: array2c
    complex,dimension(size(array2c,1),size(array2c,2),size(array1r)) :: &
         array2c_x_array1r
    integer  :: n, nz

    nz = size(array1r)

    do n = 1,nz
       array2c_x_array1r(:,:,n) = array1r(n)*array2c
    enddo
  end function array2c_x_array1r

  !************************ ADD *********************************************

  function array1r_p_array2r(array1r,array2r)
    ! Assumes array2r given in 2d plane and array1r given on z (dim 3)

    use io_tools, only: Message

    real,dimension(:),intent(in)        :: array1r
    real,dimension(:,:),intent(in)      :: array2r
    real,dimension(size(array2r,1),size(array2r,2),size(array1r)) :: &
         array1r_p_array2r
    integer :: n, nz

    nz = size(array1r)
    do n = 1,nz
       array1r_p_array2r(:,:,n) = array1r(n) + array2r
    enddo

  end function array1r_p_array2r

  !*************

  function array2r_p_array1r(array2r,array1r)
    ! Assumes array2r given in 2d plane and array1r given on z (dim 3)

    use io_tools, only: Message

    real,dimension(:),intent(in)        :: array1r
    real,dimension(:,:),intent(in)      :: array2r
    real,dimension(size(array2r,1),size(array2r,2),size(array1r)) :: &
         array2r_p_array1r
    integer :: n, nz

    nz = size(array1r)
    do n = 1,nz
       array2r_p_array1r(:,:,n) = array2r + array1r(n)
    enddo

  end function array2r_p_array1r

  !*************

  function array2c_p_array3c(array2c,array3c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to add each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    complex,dimension(:,:),intent(in)     :: array2c
    complex,dimension(:,:,:),intent(in)   :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array2c_p_array3c
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array2c_p_array3c(:,:,n) = array2c+array3c(:,:,n)
    enddo
  end function array2c_p_array3c

  !*************

  function array3c_p_array2c(array3c,array2c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to add each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    complex,dimension(:,:),intent(in)     :: array2c
    complex,dimension(:,:,:),intent(in)   :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array3c_p_array2c
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array3c_p_array2c(:,:,n) = array2c+array3c(:,:,n)
    enddo

  end function array3c_p_array2c

  !***************************** SUBTRACT **********************************

  function array1r_m_array2r(array1r,array2r)
    ! Assumes array2r given in 2d plane and array1r given on z (dim 3)

    use io_tools, only: Message

    real,dimension(:),intent(in)        :: array1r
    real,dimension(:,:),intent(in)      :: array2r
    real,dimension(size(array2r,1),size(array2r,2),size(array1r)) :: &
         array1r_m_array2r
    integer :: n, nz

    nz = size(array1r)
    do n = 1,nz
       array1r_m_array2r(:,:,n) = array1r(n) - array2r
    enddo

  end function array1r_m_array2r

  !*************

  function array2r_m_array1r(array2r,array1r)
    ! Assumes array2r given in 2d plane and array1r given on z (dim 3)

    use io_tools, only: Message

    real,dimension(:),intent(in)        :: array1r
    real,dimension(:,:),intent(in)      :: array2r
    real,dimension(size(array2r,1),size(array2r,2),size(array1r)) :: &
         array2r_m_array1r
    integer :: n, nz

    nz = size(array1r)
    do n = 1,nz
       array2r_m_array1r(:,:,n) = array2r - array1r(n)
    enddo

  end function array2r_m_array1r

  !*************

  function array2c_m_array3c(array2c,array3c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to subtract each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    complex,dimension(:,:),intent(in)     :: array2c
    complex,dimension(:,:,:),intent(in)   :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array2c_m_array3c
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array2c_m_array3c(:,:,n) = array2c-array3c(:,:,n)
    enddo
  end function array2c_m_array3c

  !*************

  function array3c_m_array2c(array3c,array2c)
    ! Assumes array2r is some fixed real field in the 2d (x-y or kx-ky) plane
    ! by which we want to subtract each level (dim 3) of complex 3d array 

    use io_tools, only: Message

    complex,dimension(:,:),intent(in)     :: array2c
    complex,dimension(:,:,:),intent(in)   :: array3c
    complex,dimension(size(array3c,1),size(array3c,2),size(array3c,3)) :: &
         array3c_m_array2c
    integer  :: n, nz

    nz = size(array3c,3)
    do n = 1,nz
       array3c_m_array2c(:,:,n) = array3c(:,:,n)-array2c
    enddo
  end function array3c_m_array2c

  !*************

end module op_rules
