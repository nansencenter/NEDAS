program main_perturbation

    use netcdf
    use module_pseudo2d
    use module_random_field

    implicit none

    integer :: i_step, i, id, file_exist
    integer :: ncid, stat, x_dimid, y_dimid
    integer, allocatable, dimension(:) :: varid
    real(8), allocatable, dimension(:,:,:) :: synforc
    character(150) :: FILE_NAME, fileID

    call read_nml() !read settings from pseudo2d.nml

    allocate(varid(n_field))
    allocate(synforc(idm,jdm,n_field))
    synforc=0.

    do i_step = 0,n_sample
        call rand_update(synforc, i_step)

        write(fileID,'(I4.4)') i_step
        FILE_NAME='synforc_'//trim(adjustl(fileID))//'.nc'
        stat = nf90_create(FILE_NAME, NF90_CLOBBER, ncid)
        stat = nf90_def_dim(ncid, 'x', xdim, x_dimid)
        stat = nf90_def_dim(ncid, 'y', ydim, y_dimid)
        do i=1,n_field
            stat = nf90_def_var(ncid, field(i)%name, NF90_FLOAT, (/ x_dimid, y_dimid /), varid(i))
        end do
        stat = nf90_enddef(ncid)
        do i = 1, n_field
            stat = nf90_put_var(ncid, varid(i), synforc(1:xdim,1:ydim,i))
        enddo
        stat =  nf90_close(ncid)
    enddo

    deallocate(varid)
    deallocate(synforc)

end program
