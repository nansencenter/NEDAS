program main_perturbation

    use netcdf
    use module_pseudo2d
    use module_random_field

    implicit none

    integer :: i_step, i, id, m, ix, jy
    integer :: ncid, stat, x_dimid, y_dimid, m_id, t_id
    integer, allocatable, dimension(:) :: varid
    real(8), allocatable, dimension(:,:,:,:) :: synforc
    character(150) :: FILE_NAME, fileID
    logical :: lognormal

    call read_nml() !read settings from perturbation.nml

    allocate(varid(n_field))
    allocate(synforc(idm,jdm,nens,n_field))
    synforc=0.

    do i_step = 0,n_sample
        if(debug) print *, i_step
        call rand_update(synforc, i_step)

        write(fileID,'(I4.4)') i_step
        FILE_NAME='synforc_'//trim(adjustl(fileID))//'.nc'
        stat = nf90_create(FILE_NAME, NF90_CLOBBER, ncid)
        stat = nf90_def_dim(ncid, 'x', xdim, x_dimid)
        stat = nf90_def_dim(ncid, 'y', ydim, y_dimid)
        stat = nf90_def_dim(ncid, 'member', nens, m_id)
        stat = nf90_def_dim(ncid, 'time', 1, t_id)
        do i=1,n_field
            stat = nf90_def_var(ncid, field(i)%name, NF90_FLOAT, (/ x_dimid, y_dimid, m_id, t_id /), varid(i))
        end do
        stat = nf90_enddef(ncid)
        do i = 1, n_field
            !!some variables that need special treatment
            ! restricted between variable bounds
            !if (field(i)%name .eq. 'dwlongw ') synforc(:,:,:,i) = max(synforc(:,:,:,i), 0.)
            !if (field(i)%name .eq. 'clouds  ') synforc(:,:,:,i) = min(max(synforc(:,:,:,i), 0.), 1.)

            !lognormal format, note in the original code for TOPAZ,
            !exp term is multiplied to the corresponding field.
            !the -0.5*var^2 term is a bias correction.
            lognormal=.false.
            if (field(i)%name .eq. 'snowfall') lognormal=.true.
            if (field(i)%name .eq. 'precip  ') lognormal=.true.

            !!save the field to synforc
            if (lognormal) then
                stat = nf90_put_var(ncid, varid(i), exp(synforc(1:xdim,1:ydim,1:nens,i) - 0.5*field(i)%vars**2))
            else
                stat = nf90_put_var(ncid, varid(i), synforc(1:xdim,1:ydim,1:nens,i))
            end if

        enddo
        stat =  nf90_close(ncid)
    enddo

    deallocate(varid)
    deallocate(synforc)

end program
