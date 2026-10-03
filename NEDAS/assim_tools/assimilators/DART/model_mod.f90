! DART model_mod for NEDAS: no model of its own, the state metadata come from NEDAS.
!
! Each task holds one NEDAS block of nloc state entries, padded to nmax. Global index
! g = (k-1)*ntasks + task + 1, so DART's round-robin ownership is exactly that block.
module model_mod

use types_mod,            only : r8, i8
use time_manager_mod,     only : time_type, set_time
use location_mod,         only : location_type, set_location, get_close_obs, get_close_state, &
                                 set_periodic
use utilities_mod,        only : error_handler, E_ERR
use netcdf_utilities_mod, only : nc_begin_define_mode, nc_end_define_mode
use state_structure_mod,  only : add_domain
use ensemble_manager_mod, only : ensemble_type
use mpi_utilities_mod,    only : my_task_id, task_count
use default_model_mod,    only : pert_model_copies, read_model_time, write_model_time, &
                                 init_time => fail_init_time, &
                                 init_conditions => fail_init_conditions, &
                                 convert_vertical_obs, convert_vertical_state, adv_1step

implicit none
private

public :: get_model_size, get_state_meta_data, model_interpolate, end_model, &
          static_init_model, nc_write_model_atts, get_close_obs, get_close_state, &
          pert_model_copies, convert_vertical_obs, convert_vertical_state, &
          read_model_time, adv_1step, init_time, init_conditions, &
          shortest_time_between_assimilations, write_model_time

! set by NEDAS before filter_main
public :: nedas_set_state_meta, nedas_set_periodic, nedas_nloc, nedas_nmax

character(len=*), parameter :: source = 'nedas model_mod.f90'

integer :: nedas_nloc = 0, nedas_nmax = 0
real(r8), allocatable :: sx(:), sy(:), sz(:)
integer,  allocatable :: sqty(:)
logical :: domain_added = .false.
! periodic axes; the location_nml flags alone do not enable periodic distances
logical  :: px = .false., py = .false.
real(r8) :: pxmin, pxmax, pymin, pymax

contains

subroutine nedas_set_state_meta(nloc, nmax, x, y, z, qty)
integer,  intent(in) :: nloc, nmax
real(r8), intent(in) :: x(nloc), y(nloc), z(nloc)
integer,  intent(in) :: qty(nloc)

if (allocated(sx)) deallocate(sx, sy, sz, sqty)
allocate(sx(nloc), sy(nloc), sz(nloc), sqty(nloc))
sx = x; sy = y; sz = z; sqty = qty
nedas_nloc = nloc
nedas_nmax = nmax
end subroutine nedas_set_state_meta


subroutine nedas_set_periodic(x_on, xmin, xmax, y_on, ymin, ymax)
logical,  intent(in) :: x_on, y_on
real(r8), intent(in) :: xmin, xmax, ymin, ymax
px = x_on; pxmin = xmin; pxmax = xmax
py = y_on; pymin = ymin; pymax = ymax
end subroutine nedas_set_periodic


subroutine static_init_model()
integer :: dom_id
if (domain_added) return
if (px) call set_periodic('x', pxmin, pxmax)
if (py) call set_periodic('y', pymin, pymax)
if (nedas_nmax <= 0) call error_handler(E_ERR, 'static_init_model', &
   'NEDAS state metadata not set', source)
dom_id = add_domain(get_model_size())
domain_added = .true.
end subroutine static_init_model


function get_model_size()
integer(i8) :: get_model_size
get_model_size = int(nedas_nmax, i8) * task_count()
end function get_model_size


subroutine get_state_meta_data(index_in, location, qty)
integer(i8),         intent(in)            :: index_in
type(location_type), intent(out)           :: location
integer,             intent(out), optional :: qty

integer :: k

if (mod(index_in - 1, int(task_count(), i8)) /= my_task_id()) &
   call error_handler(E_ERR, 'get_state_meta_data', 'index not owned by this task', source)
k = int((index_in - 1) / task_count()) + 1

if (k <= nedas_nloc) then
   location = set_location(sx(k), sy(k), sz(k))
   if (present(qty)) qty = sqty(k)
else if (nedas_nloc > 0) then   ! padding: constant, so never updated
   location = set_location(sx(1), sy(1), sz(1))
   if (present(qty)) qty = sqty(1)
else
   location = set_location(0.0_r8, 0.0_r8, 0.0_r8)
   if (present(qty)) qty = 1
endif
end subroutine get_state_meta_data


! H(x) always comes from NEDAS (precomputed forward operators)
subroutine model_interpolate(state_handle, ens_size, location, qty, expected_obs, istatus)
type(ensemble_type), intent(in)  :: state_handle
integer,             intent(in)  :: ens_size
type(location_type), intent(in)  :: location
integer,             intent(in)  :: qty
real(r8),            intent(out) :: expected_obs(ens_size)
integer,             intent(out) :: istatus(ens_size)
expected_obs = 0.0_r8
istatus = 1
end subroutine model_interpolate


! obs window is set by NEDAS; this only has to cover it
function shortest_time_between_assimilations()
type(time_type) :: shortest_time_between_assimilations
shortest_time_between_assimilations = set_time(0, 3650)
end function shortest_time_between_assimilations


subroutine nc_write_model_atts(ncid, domain_id)
integer, intent(in) :: ncid, domain_id
call nc_begin_define_mode(ncid)
call nc_end_define_mode(ncid)
end subroutine nc_write_model_atts


subroutine end_model()
end subroutine end_model

end module model_mod
