module syscalls           

  ! Contains generic wrappers for command line argument getter (eg GETARG)
  !
  ! This one is for Linux.
  !
  ! On Cray T90 machines, PXFGETARG gets the command line argument, 
  ! but on almost any other system its GETARG.  Man pages usually
  ! have entries for IARGC and GETARG, as well as IS_NAN (which
  ! is also the typical NAN tester on other systems).

  implicit none

contains

  subroutine get_arg(narg,string,nchars,iocheck)

    integer,intent(in)       :: narg
    character(*),intent(out) :: string
    integer,intent(out)      :: nchars
    integer,external         :: iargc
    integer                  :: iocheck

    iocheck = 0  ! No io status from this particular call

    if (iargc()>narg-1) then
       call getarg(narg,string)
       nchars = len_trim(string)
    else
       return  ! Do nothing if there is no narg_th argument
    endif

  end subroutine get_arg

end module syscalls
