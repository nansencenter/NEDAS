module io_tools                   !-*-f90-*-

  !**************************************************************************
  ! This module contains base IO routines for unformatted, direct access
  ! files.  It is designed for use with Matlab using the SQGM routines
  ! (available wherever you got this from). 
  !
  ! Dependencies: None, *except* you need to know the default record width
  !               for your compiler/system.  Read #2 under Pass_params
  !               carefully.
  !
  ! ROUTINES:
  !
  ! Pass_params(datadirin,recunitin):  Call to set internal values of 
  !   two parameters, datadir and recunit, necessary for these 
  !   routines to work. 
  !   If this routine is not called, these two variables will default to
  !   '\.' and 8, respectively.  Here is what they are for:
  !
  !   1. datadir:  sets the directory to which all data is written. 
  !
  !   2. recunit:  used in setting the record length for
  !     direct access writes and reads.  In particular, the routine Open_file
  !     opens files of width 'rec_width' (an argument to the routine), which
  !     is essentially the width of the record in number of variable values.
  !     i.e. if you have a field f=f(1:nx,1:ny), and you want to make the
  !     file record 'nx' wide, then rec_width=nx.  In order to let 'rec_width'
  !     be just the number of variables in a record, we must have a conversion
  !     factor to whatever unit the compiler counts records in.  That is what
  !     'recunit' is.  So, if the compiler counts units in bytes (as the
  !     SGI or Cray will with the -bytereclen compiler option), and if you are
  !     using 8 byte real variables (REAL*8, or compiler option -r8), then
  !     we must have recunit = 8.  For DEC compilers/computers, records 
  !     are counted in 4-byte units, so with REAL*8 variables, we need 
  !     recunit=2.
  !  
  ! Message(msg,tag,fatal,r_tag):  Writes message 'msg' to standard out and to 
  !     log file in datadir (defined by 'logfile' below in global module 
  !     declarations). 'tag' is an optional integer number which will be 
  !     written after message.
  !     'r_tag' is an optional real number which will be written after message.
  !     If fatal is present, no matter what its value (its a string), then
  !     program will exit -- this is for use as an error handler.  'tag'
  !     should be set to the IOSTAT return, available from any f90 I/O call.
  !     One can then use the value to figure out the exact, possibly machine 
  !     dependent, error (for example, on Cray and SGI, type 'explain lib-#',
  !     where # is the number returned by IOSTAT).
  !
  ! Open_file(fid,fnamein,stat,rec_width,exclude_dd):  For use internally to
  !     open direct access unformated files.  'fid' is the file ID, 'fnamein'
  !     is the name of the file.  If optional 'exclude_dd' is included (no 
  !     matter what its value -- its an integer), then 'datadir' (defined
  !     globally for this module) will be prepended to 'fnamein'.  'stat'
  !     is the file status, i.e. 'old', 'new', 'unknown', etc (see f90 doc
  !     on OPEN).  'rec_width' is explained above.
  !
  ! Write_field(field,fname,frame):  Write 'frame' (an integer counter) of
  !     array 'field' to file 'fname' (always placed in 'datadir', and 
  !     postpended with the extension '.bin').  'field' can be a real array
  !     of rank 0, 1, 2 or 3, or a complex array of rank 2 or 3.  Generically
  !     calls one of six routines depending on rank and type of field.
  !     Values are stored with record lengths equal to the width of the 
  !     first non-singleton dimension of 'field' (i.e. the width in x dir'n),
  !     or to 1 if field is rank 0 or 1.  Records are ordered as stacks of all
  !     dimension 2 values (i.e. the y direction values if its rank >= 2)
  !     then by dimension 3 (i.e. the z direction values if its rank = 3).
  !     If field is complex, then real part is written first, as a frame,
  !     followed by imaginary part frame.  If 'frame' is present, it sets file
  !     position to offset by (field size)*(frame-1).  Offset is 0 if frame is
  !     not present.
  !
  ! Read_field(field,fname,frame,exclude_dd):  Same as for Write_field, only
  !    it reads from 'fname'.  Additionally, optional 'exclude_dd' will, if
  !    present (no matter what its value -- its type is integer), 
  !    imply that 'fname'.bin contains path for file.  Otherwise it looks 
  !    for 'fname' in 'datadir' (default).
  !
  !**************************************************************************

  implicit none
  private

  integer                 :: iocheck, fida=50
  integer,save            :: recunit=8       ! sgi => 8, dec => 2
  character(50),save      :: datadir='./', runlog='run.log'

  public :: Read_field, Write_field, Message, Pass_params, Open_file

  ! Generic procedures for 1d, 2d and 3d fields, real or complex

  interface Read_field
     module procedure Read_value, Read_array, Read_field2, &
          Read_field2c, Read_field3, Read_field3c
  end interface
  interface Write_field
     module procedure Write_value, Write_array, Write_field2, &
          Write_field2c, Write_field3, Write_field3c
  end interface

  !************************************************************************

contains

  subroutine Pass_params(datadirin,recunitin)
    character(50),intent(in)      :: datadirin
    integer,intent(in)            :: recunitin
    datadir = datadirin
    recunit = recunitin
  end subroutine Pass_params

  subroutine Message(msg,tag,fatal,r_tag)
    character(*),intent(in)               :: msg
    integer,intent(in),optional           :: tag
    character(*),intent(in),optional      :: fatal
    real,intent(in),optional              :: r_tag
    open(unit=fida,file=trim(datadir)//trim(runlog),position='append')
    if ((.not.present(tag)).and.(.not.present(r_tag))) then
       write(fida,*) msg
       print*, msg
    elseif (present(tag).and..not.present(r_tag)) then
       write(fida,*) msg,tag
       print*, msg,tag
    elseif (present(r_tag).and..not.present(tag)) then
       write(fida,*) msg,r_tag
       print*, msg,r_tag
    elseif (present(tag).and.present(r_tag)) then
       write(fida,*) msg,tag,r_tag
       print*, msg,tag,r_tag
    endif
    close(fida)
    if (present(fatal)) stop
  end subroutine Message

  subroutine Open_file(fid,fnamein,stat,rec_width,exclude_dd)
    integer               :: fid,rec_width
    character(*)          :: fnamein,stat
    integer,optional      :: exclude_dd
    character(80)         :: fnameo
    logical               :: file_exists
    if (present(exclude_dd)) then    ! By default, expect to open in datadir
       fnameo = trim(fnamein)//'.bin'
    else
       fnameo = trim(datadir)//trim(fnamein)//'.bin'
    endif
    inquire(file=fnameo,exist=file_exists)
    if ((.not.file_exists).and.(stat=='old')) then
       call Message('Specified file is not present: '//fnameo,fatal='y')
    endif
    open(unit=fid,file=fnameo,status=stat,form='unformatted',&
         access='direct',recl=recunit*rec_width,iostat=iocheck)
    if (iocheck/=0) call Message('Open_file error, name & iocheck: '&
         &//fnameo,iocheck,'y')
  end subroutine Open_file

  subroutine Write_value(field,fname,frame)
    real                          :: field
    character(*),intent(in)       :: fname
    integer,optional,intent(in)   :: frame
    integer                       :: offset=0
    call Open_file(fida,fname,'unknown',1)
    if (present(frame)) offset = frame-1
    write(unit=fida,rec=1+offset,iostat=iocheck) field
    if (iocheck/=0) call Message('Write_value error, name & iocheck: '&
         &//fname,tag=iocheck,fatal='y')
    close(fida)
  end subroutine Write_value

  subroutine Read_value(field,fname,frame,exclude_dd)
    real                          :: field
    character(*),intent(in)       :: fname
    integer,optional,intent(in)   :: frame, exclude_dd
    integer                       :: offset=0
    if (present(exclude_dd)) then
       call Open_file(fida,fname,'old',1,exclude_dd)
    else
       call Open_file(fida,fname,'old',1)
    endif
    if (present(frame)) offset = frame-1
    read(unit=fida,rec=1+offset,iostat=iocheck) field
    if (iocheck/=0) call Message('Read_value error, name & iocheck: '&
         &//fname,iocheck,'y')
    close(fida)
  end subroutine Read_value

  subroutine Write_array(field,fname,frame)
    real,dimension(:)             :: field
    character(*),intent(in)       :: fname
    integer,optional,intent(in)   :: frame
    integer                       :: n,j, offset=0
    n = size(field)
    call Open_file(fida,fname,'unknown',1)
    if (present(frame)) offset = (frame-1)*n
    do j = 1,n
       write(unit=fida,rec=offset+j,iostat=iocheck) field(j)
       if (iocheck/=0) call Message('Write_array error, name & iocheck: '&
            &//fname,iocheck,'y')
    enddo
    close(fida)
  end subroutine Write_array

  subroutine Read_array(field,fname,frame,exclude_dd)
    real,dimension(:)             :: field
    character(*),intent(in)       :: fname
    integer,optional,intent(in)   :: frame, exclude_dd
    integer                       :: n,j, offset=0
    n = size(field)
    if (present(exclude_dd)) then
       call Open_file(fida,fname,'old',1,exclude_dd)
    else
       call Open_file(fida,fname,'old',1)
    endif
    if (present(frame)) offset = (frame-1)*n
    do j = 1,n
       read(unit=fida,rec=j+offset,iostat=iocheck) field(j)
       if (iocheck/=0) call Message('Read_array error, name & iocheck: '&
            &//fname,iocheck,'y')
    enddo
    close(fida)
  end subroutine Read_array

  subroutine Write_field2(field,fname,frame)
    real,dimension(:,:),intent(in)        :: field
    character(*),intent(in)               :: fname
    integer,optional,intent(in)           :: frame
    integer                               :: nx,ny,i,j,offset=0
    nx = size(field,1); ny = size(field,2)
    call Open_file(fida,fname,'unknown',nx)
    if (present(frame)) offset = ny*(frame-1)
    do j=1,ny
       write(fida,rec=j+offset,iostat=iocheck)(field(i,j),i=1,nx)
       if (iocheck/=0) call Message('Write_field2 error, name & iocheck: '&
            &//fname,iocheck,'y')
    enddo
    close(fida)
  end subroutine Write_field2

  subroutine Read_field2(field,fname,frame,exclude_dd)
    real,dimension(:,:),intent(out)       :: field
    character(*),intent(in)               :: fname
    integer,optional,intent(in)           :: frame, exclude_dd
    integer                               :: i,j,nx,ny,offset=0
    nx = size(field,1); ny = size(field,2)
    if (present(exclude_dd)) then
       call Open_file(fida,fname,'old',nx,exclude_dd)
    else
       call Open_file(fida,fname,'old',nx)
    endif
    if (present(frame)) offset = ny*(frame-1)
    do j = 1,ny
       read(fida,rec=j+offset,iostat=iocheck)(field(i,j),i=1,nx)
       if (iocheck/=0) call Message('Read_field2 error, name & iocheck: '&
            &//fname,iocheck,'y')
    enddo
    close(fida)
  end subroutine Read_field2

  subroutine Write_field2c(field,fname,frame)
    complex,dimension(:,:),intent(in)     :: field
    character(*),intent(in)               :: fname
    integer,optional,intent(in)           :: frame
    integer                               :: nx,ny,i,j,offsetr,offseti
    real,dimension(size(field,1),size(field,2)) :: fr, fi
    nx = size(field,1); ny = size(field,2)
    fr = real(field); fi = aimag(field)
    call Open_file(fida,fname,'unknown',nx)
    offsetr = 0; offseti = ny
    if (present(frame)) then
       offsetr = ny*2*(frame-1)
       offseti = ny*(2*(frame-1)+1)
    endif
    do j=1,ny
       write(fida,rec=j+offsetr,iostat=iocheck)(fr(i,j),i=1,nx)
       if (iocheck/=0) call Message('Write_field2c error, name & iocheck: '&
            &//fname,iocheck,'y')
       write(fida,rec=j+offseti,iostat=iocheck)(fi(i,j),i=1,nx)
       if (iocheck/=0) call Message('Write_field2c error, name & iocheck: '&
            &//fname,iocheck,'y')
    enddo
    close(fida)
  end subroutine Write_field2c

  subroutine Read_field2c(field,fname,frame,exclude_dd)
    complex,dimension(:,:),intent(out)    :: field
    character(*),intent(in)               :: fname
    integer,optional,intent(in)           :: frame, exclude_dd
    integer                               :: i,j,nx,ny,offsetr,offseti
    real,dimension(size(field,1),size(field,2)) :: fr, fi
    nx = size(field,1); ny = size(field,2)
    if (present(exclude_dd)) then
       call Open_file(fida,fname,'old',nx,exclude_dd)
    else
       call Open_file(fida,fname,'old',nx)
    endif
    offsetr = 0; offseti = ny
    if (present(frame)) then
       offsetr = ny*2*(frame-1)
       offseti = ny*(2*(frame-1)+1)
    endif
    do j=1,ny
       read(fida,rec=j+offsetr,iostat=iocheck) (fr(i,j),i=1,nx)
       if (iocheck/=0) call Message('Read_field2c error, name & iocheck: '&
            &//fname,iocheck,'y')
       read(fida,rec=j+offseti,iostat=iocheck) (fi(i,j),i=1,nx)
       if (iocheck/=0) call Message('Read_field2c error, name & iocheck: '&
            &//fname,iocheck,'y')
    enddo
    field=cmplx(fr,fi)
    close(fida)
  end subroutine Read_field2c

  subroutine Write_field3(field,fname,frame)
    real,dimension(:,:,:),intent(in)      :: field
    character(*),intent(in)               :: fname
    integer,optional,intent(in)           :: frame
    integer                               :: nx,ny,nz,i,j,k,offset=0
    nx = size(field,1); ny = size(field,2); nz = size(field,3)
    call Open_file(fida,fname,'unknown',nx)
    if (present(frame)) offset = ny*nz*(frame-1)
    do k=1,nz
       do j=1,ny
          write(fida,rec=(k-1)*ny+j+offset,iostat=iocheck)&
               (field(i,j,k),i=1,nx)
          if (iocheck/=0) call Message('Write_field3 error, name & iocheck: '&
               &//fname,iocheck,'y')
       enddo
    enddo
    close(fida)
  end subroutine Write_field3

  subroutine Read_field3(field,fname,frame,exclude_dd)
    real,dimension(:,:,:),intent(out)     :: field
    character(*),intent(in)               :: fname
    integer,optional,intent(in)           :: frame,exclude_dd
    integer                               :: i,j,k,nx,ny,nz,offset=0
    nx = size(field,1); ny = size(field,2); nz = size(field,3)
    if (present(exclude_dd)) then
       call Open_file(fida,fname,'old',nx,exclude_dd)
    else
       call Open_file(fida,fname,'old',nx)
    endif
    if (present(frame)) offset = ny*nz*(frame-1)
    do k=1,nz
       do j=1,ny
          read(fida,rec=(k-1)*ny+j+offset,iostat=iocheck) &
               (field(i,j,k),i=1,nx)
          if (iocheck/=0) call Message('Read_field3c error, name & iocheck: '&
               &//fname,iocheck,'y')
       enddo
    enddo
    close(fida)
  end subroutine Read_field3

  subroutine Write_field3c(field,fname,frame)
    complex,dimension(:,:,:),intent(in)   :: field
    character(*),intent(in)               :: fname
    integer,optional,intent(in)           :: frame
    integer                               :: nx,ny,nz,i,j,k,offsetr,offseti
    real,dimension(size(field,1),size(field,2),size(field,3)) :: fr, fi
    nx = size(field,1); ny = size(field,2); nz = size(field,3)
    fr = real(field); fi = aimag(field)
    call Open_file(fida,fname,'unknown',nx)
    offsetr = 0; offseti = ny*nz
    if (present(frame)) then
       offsetr = ny*nz*2*(frame-1)
       offseti = ny*nz*(2*(frame-1)+1)
    endif
    do k=1,nz
       do j=1,ny
          write(fida,rec=(k-1)*ny+j+offsetr,iostat=iocheck)&
               (fr(i,j,k),i=1,nx)
          if (iocheck/=0) call Message('Write_field3c err, name & iocheck: '&
               &//fname,iocheck,'y')
          write(fida,rec=(k-1)*ny+j+offseti,iostat=iocheck)&
               (fi(i,j,k),i=1,nx)
          if (iocheck/=0) call Message('Write_field3c err, name & iocheck: '&
               &//fname,iocheck,'y')
       enddo
    enddo
    close(fida)
  end subroutine Write_field3c

  subroutine Read_field3c(field,fname,frame,exclude_dd)
    complex,dimension(:,:,:),intent(out)  :: field
    character(*),optional,intent(in)      :: fname
    integer,optional,intent(in)           :: frame,exclude_dd
    integer                               :: i,j,k,nx,ny,nz,offsetr,offseti
    real,dimension(size(field,1),size(field,2),size(field,3)) :: fr, fi
    nx = size(field,1); ny = size(field,2); nz = size(field,3)
    if (present(exclude_dd)) then
       call Open_file(fida,trim(fname),'old',nx,exclude_dd)
    else
       call Open_file(fida,trim(fname),'old',nx)
    endif
    offsetr = 0; offseti = ny*nz
    if (present(frame)) then
       offsetr = ny*nz*2*(frame-1)
       offseti = ny*nz*(2*(frame-1)+1)
    endif
    do k=1,nz
       do j=1,ny
          read(fida,rec=(k-1)*ny+j+offsetr,iostat=iocheck) &
               (fr(i,j,k),i=1,nx)
          if (iocheck/=0) call Message('Read_field3c error, name & iocheck: '&
               &//fname,iocheck,'y')
          read(fida,rec=(k-1)*ny+j+offseti,iostat=iocheck) &
               (fi(i,j,k),i=1,nx)
          if (iocheck/=0) call Message('Read_field3c error, name & iocheck: '&
               &//fname,iocheck,'y')
       enddo
    enddo
    field=cmplx(fr,fi)
    close(fida)
  end subroutine Read_field3c

end module io_tools
