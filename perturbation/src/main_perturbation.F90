program main_perturbation

    use netcdf
    use module_pseudo2d
    use module_random_field

    implicit none

    integer :: i, id, m, ix, jy
    integer :: ncid, ncid_prev, stat, x_dimid, y_dimid, m_id, t_id
    integer, allocatable, dimension(:) :: varid, varid_prev
    real(8), allocatable, dimension(:,:,:,:) :: perturb, perturb_prev
    character(150) :: FILE_NAME, fileID

    call read_nml() !read settings from perturbation.nml

    allocate(varid(n_field))
    allocate(varid_prev(n_field))
    allocate(perturb(idm,jdm,nens,n_field))
    allocate(perturb_prev(idm,jdm,nens,n_field))

    if(debug) print *, 'generating sample: ', i_sample

    !!!read previous perturb
    if(i_sample>0) then
        write(fileID,'(I4.4)') (i_sample-1)
        FILE_NAME='perturb_'//trim(adjustl(fileID))//'.nc'
        stat = nf90_open(FILE_NAME, NF90_NOWRITE, ncid_prev)
        if(stat>0) stop 'previous perturb file not found'
        do i=1,n_field
            stat = nf90_inq_varid(ncid_prev, field(i)%name, varid_prev(i))
        end do
        do i=1,n_field
            stat = nf90_get_var(ncid_prev, varid_prev(i), perturb_prev(1:xdim,1:ydim,1:nens,i))
        enddo
        stat =  nf90_close(ncid_prev)
    endif

    !!!generate and update random field
    call rand_update(perturb, i_sample, perturb_prev)

    !!!write the random field to perturb.nc
    write(fileID,'(I4.4)') i_sample
    FILE_NAME='perturb_'//trim(adjustl(fileID))//'.nc'
    stat = nf90_create(FILE_NAME, NF90_CLOBBER, ncid)
    stat = nf90_def_dim(ncid, 'x', xdim, x_dimid)
    stat = nf90_def_dim(ncid, 'y', ydim, y_dimid)
    stat = nf90_def_dim(ncid, 'member', nens, m_id)
    stat = nf90_def_dim(ncid, 'time', 1, t_id)
    do i=1,n_field
        stat = nf90_def_var(ncid, field(i)%name, NF90_FLOAT, (/ x_dimid, y_dimid, m_id, t_id /), varid(i))
    end do
    stat = nf90_enddef(ncid)
    do i=1,n_field
        stat = nf90_put_var(ncid, varid(i), perturb(1:xdim,1:ydim,1:nens,i))
    enddo
    stat =  nf90_close(ncid)

    deallocate(varid, varid_prev)
    deallocate(perturb, perturb_prev)

end program
