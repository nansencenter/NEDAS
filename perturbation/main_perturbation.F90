program main_perturbation

    use netcdf
    use module_pseudo2d
    use module_random_field

    implicit none

    !!todo: make input x, y flexible, but then=> 2^n nearest larger than x,y,
    !afterwards trim to size correctly.
    integer, parameter:: xdim = 1024, ydim = 1024
    real(8) :: synforc(xdim, ydim, 6)
    integer :: i_step, i, id, file_exist
    integer :: ncid, stat, x_dimid, y_dimid
    integer :: varid(6)
    character(150) :: FILE_NAME, fileID

    call limits_randf(xdim, ydim)  ! read in setting from pseudo2d.nml

    do i_step = 0,100  !!!??
        synforc=0.
        !-------------------------------
        call init_fvars                ! init field variables
        call init_rand_update(synforc)
        ! synforc is pure output overwrite everytime
        ! i_step=1 indicates to create a perturbation for the previous time.
        !-------------------------------

        write(fileID,'(I4)') i_step
        FILE_NAME='synforc_'//trim(adjustl(fileID))//'.nc'
        ! Create the netCDF file. The nf90_clobber parameter tells netCDF to overwrite this file, if it already exists.
        stat = nf90_create(FILE_NAME, NF90_CLOBBER, ncid)
        ! Define the coordinate dimensions
        stat = nf90_def_dim(ncid, 'x', xdim, x_dimid)
        stat = nf90_def_dim(ncid, 'y', ydim, y_dimid)
        ! Define the variable.
        stat = nf90_def_var(ncid, "uwind",    NF90_FLOAT, (/ x_dimid, y_dimid /), varid(1))
        stat = nf90_def_var(ncid, "vwind",    NF90_FLOAT, (/ x_dimid, y_dimid /), varid(2))
        stat = nf90_def_var(ncid, "snowfall", NF90_FLOAT, (/ x_dimid, y_dimid /), varid(3))
        stat = nf90_def_var(ncid, "Qlw_in",   NF90_FLOAT, (/ x_dimid, y_dimid /), varid(4))
        stat = nf90_def_var(ncid, "sss",      NF90_FLOAT, (/ x_dimid, y_dimid /), varid(5))
        stat = nf90_def_var(ncid, "sst",      NF90_FLOAT, (/ x_dimid, y_dimid /), varid(6))
        ! End define mode. This tells netCDF we are done defining metadata.
        stat = nf90_enddef(ncid)
        ! Write the pretend data to the file. Although netCDF supports
        ! reading and writing subsets of data, in this case we write all the
        ! data in one operation.
        do i = 1, 6
            stat = nf90_put_var(ncid, varid(i), synforc(:,:,i))
        enddo
        ! Close the file. This frees up any internal netCDF resources associated with the file, and flushes any buffers.
        stat =  nf90_close(ncid)

    enddo

end program
