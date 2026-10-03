! C-callable interface to DART's bgrid_solo model, for running it in memory (NEDAS online io mode).
!
! The state is DART's state vector: ps(nlon,nlat), t(nlon,nlat,nlev), u(nlon,nlat-1,nlev),
! v(nlon,nlat-1,nlev), in this order (model_nml state_variables) and Fortran storage order, the
! same layout as the variables in a DART restart file. Advancing it calls adv_1step in the same
! loop as DART's advance_state (used by nedas_bgrid_advance), so the two give identical states.
!
! The model is configured by input.nml in the current directory when nedas_bgrid_init is
! called (DART modules read their namelists from there); model_nml%template_file is 'null', so
! no file is needed for the state layout. The model has a single static state (FMS module
! variables), so a process holds one configuration, and members are advanced in turn.
module nedas_bgrid_lib

use iso_c_binding,     only : c_int, c_double, c_long_long
use types_mod,         only : r8, i8
use time_manager_mod,  only : time_type, set_time, get_time, operator(<), operator(+)
use mpi_utilities_mod, only : initialize_mpi_utilities
use assim_model_mod,   only : static_init_assim_model, get_model_size, get_model_time_step
use model_mod,         only : init_conditions, adv_1step

implicit none
private

logical :: initialized = .false.

contains

! read input.nml in the current directory and set up the model; returns the state vector size
subroutine nedas_bgrid_init(model_size) bind(C, name='nedas_bgrid_init')
integer(c_long_long), intent(out) :: model_size
if (.not. initialized) then
   call initialize_mpi_utilities('nedas_bgrid_lib')
   call static_init_assim_model()
   initialized = .true.
endif
model_size = int(get_model_size(), c_long_long)
end subroutine nedas_bgrid_init

! model time step in seconds
subroutine nedas_bgrid_time_step(seconds) bind(C, name='nedas_bgrid_time_step')
integer(c_int), intent(out) :: seconds
integer :: s, d
call get_time(get_model_time_step(), s, d)
seconds = int(s + 86400 * d, c_int)
end subroutine nedas_bgrid_time_step

! the model's cold start: at rest, with the Held-Suarez equilibrium temperature structure
subroutine nedas_bgrid_cold_start(n, x) bind(C, name='nedas_bgrid_cold_start')
integer(c_long_long), value       :: n
real(c_double),     intent(out)   :: x(n)
call init_conditions(x)
end subroutine nedas_bgrid_cold_start

! advance x in place from model time (days, seconds) by advance_seconds
subroutine nedas_bgrid_advance(n, x, days, seconds, advance_seconds) bind(C, name='nedas_bgrid_advance')
integer(c_long_long), value         :: n
real(c_double),     intent(inout)   :: x(n)
integer(c_int),       value         :: days, seconds, advance_seconds
type(time_type) :: model_time, target_time, time_step
real(r8), allocatable :: y(:)

model_time  = set_time(seconds, days)
target_time = model_time + set_time(mod(advance_seconds, 86400), advance_seconds / 86400)
time_step   = get_model_time_step()
allocate(y(n))  ! adv_1step takes an assumed-shape array
y = x
do while (model_time < target_time)
   call adv_1step(y, model_time)
   model_time = model_time + time_step
enddo
x = y
deallocate(y)
end subroutine nedas_bgrid_advance

end module nedas_bgrid_lib
