! C-callable wrapper around DART's serial-filter kernels, for NEDAS.
!
! NEDAS keeps ownership of the grid, partitioning, localization, obs matching and
! I/O; only the innermost update math below is DART's. Built into libdartkernels.so
! by build_dart_kernels.sh and called from core.py via ctypes.
!
! The kernels used here are private to assim_tools_mod upstream; the build script adds
! a `public ::` line for them (idempotent, `git checkout` reverts it). obs_increment_kde
! is already public in kde_distribution_mod.
!
! The filter_kind dispatch below mirrors DART's own obs_increment(), minus the
! algorithm_info_mod/QCF table lookup: NEDAS passes the choice in directly instead of
! reading it from a per-obs-type table.
module dart_kernels_mod

use, intrinsic :: iso_c_binding, only : c_int, c_double
use           types_mod,        only : r8
use           assim_tools_mod,  only : obs_increment_eakf, obs_increment_enkf,          &
                                       obs_increment_kernel, obs_increment_particle,    &
                                       obs_increment_rank_histogram, obs_increment_gamma, &
                                       obs_increment_bounded_norm_rhf,                  &
                                       get_truncated_normal_like, update_from_obs_inc,  &
                                       inc_ran_seq, first_inc_ran_call, assim_tools_init
use           random_seq_mod,   only : init_random_seq
use           mpi_utilities_mod, only : initialize_mpi_utilities
use           kde_distribution_mod, only : obs_increment_kde

implicit none
private

public :: dart_obs_increment, dart_update_from_obs_inc, dart_set_random_seed, dart_initialize

! set once dart_initialize() has run; DART warns if its init routines are called twice
logical :: dart_is_initialized = .false.

! filter kinds, mirrored in core.py's FILTER_KINDS. These are NEDAS's own codes, not
! DART's algorithm_info_mod constants, since the leaf kernels are called directly.
integer, parameter :: KIND_EAKF      = 1
integer, parameter :: KIND_ENKF      = 2
integer, parameter :: KIND_KERNEL    = 3
integer, parameter :: KIND_PARTICLE  = 4
integer, parameter :: KIND_RHF       = 5
integer, parameter :: KIND_GAMMA     = 6
integer, parameter :: KIND_BNRHF     = 7
integer, parameter :: KIND_KDE       = 8

contains

!> Bring up DART's utilities and read its namelists.
!>
!> Most kernels never need this: the wrapper otherwise avoids DART's runtime setup, which
!> is what lets EAKF reproduce NEDAS's native results with no namelist or log files in play.
!> It is needed only where a kernel's behaviour is namelist-controlled:
!>   ENKF  sort_obs_inc
!>   RHF   rectangular_quadrature, gaussian_likelihood_tails
!>   KDE   quadrature_order (read from kde_nml on first use)
!>
!> initialize_mpi_utilities is the null_mpi one (the library is built serial), so this does
!> not touch MPI; it forwards to initialize_utilities, which reads "input.nml" from the
!> current working directory. assim_tools_init() then reads &assim_tools_nml from the same
!> file -- that section is NOT optional, so the file must contain it. The caller is
!> responsible for putting a suitable input.nml in place; a missing file or section makes
!> DART stop the process. DART also writes dart_log.out/dart_log.nml there.
subroutine dart_initialize() bind(c, name='dart_initialize')

if (dart_is_initialized) return
call initialize_mpi_utilities()
call assim_tools_init()
dart_is_initialized = .true.

end subroutine dart_initialize


!> Seed the random sequence the stochastic kernels draw from.
!>
!> Without this, obs_increment_enkf and _kernel seed themselves on first use with
!> my_task_id() + 1. That has two consequences, both bad for NEDAS:
!>
!>   * my_task_id() initializes DART's (null) mpi utilities, which reads input.nml and
!>     stops the run when that file is absent.
!>   * the seed is a task id, so the stream is identical in every process. NEDAS starts a
!>     fresh process per cycle, so each cycle would replay the same perturbations.
!>
!> Seeding here sets first_inc_ran_call, so the kernels' own seeding block never runs and
!> my_task_id() is never called. The caller is responsible for passing a seed that is the
!> same on every rank (serial.py has all ranks compute the increment for one global state,
!> so they must agree) but different per cycle (so the perturbations are not replayed).
subroutine dart_set_random_seed(seed) bind(c, name='dart_set_random_seed')

integer(c_int), intent(in), value :: seed

call init_random_seq(inc_ran_seq, int(seed))
first_inc_ran_call = .false.

end subroutine dart_set_random_seed


!> Observation-space increment for one scalar obs, for the requested filter kind.
!>
!> Reproduces the guards DART's obs_increment() dispatcher applies before reaching a
!> kernel. Returns a status rather than calling error_handler, so a degenerate obs
!> cannot abort the calling python process:
!>   0 = ok
!>   1 = both obs_var and prior_var are zero (no meaningful analysis)
!>   2 = unknown filter_kind
!>   3 = likelihood underflowed (bounded normal RHF); increments left at zero
integer(c_int) function dart_obs_increment(filter_kind, ens_size, ens, obs, obs_var, &
                                           bounded_below, bounded_above,             &
                                           lower_bound, upper_bound, obs_inc, a)     &
                        bind(c, name='dart_obs_increment')

integer(c_int), intent(in), value :: filter_kind, ens_size
real(c_double), intent(in)        :: ens(ens_size)
real(c_double), intent(in), value :: obs, obs_var
integer(c_int), intent(in), value :: bounded_below, bounded_above
real(c_double), intent(in), value :: lower_bound, upper_bound
real(c_double), intent(out)       :: obs_inc(ens_size)
real(c_double), intent(out)       :: a

real(r8) :: prior_mean, prior_var, likelihood(ens_size), like_sum
logical  :: bb, ba
integer  :: i

dart_obs_increment = 0
a = 0.0_r8
obs_inc = 0.0_r8

bb = (bounded_below /= 0)
ba = (bounded_above /= 0)

prior_mean = sum(ens) / ens_size
prior_var  = sum((ens - prior_mean)**2) / (ens_size - 1)

if (obs_var == 0.0_r8 .and. prior_var == 0.0_r8) then
   dart_obs_increment = 1
   return

else if (obs_var == 0.0_r8) then
   ! delta-function obs: every member collapses onto the obs value
   obs_inc = obs - ens
   return

else if (prior_var == 0.0_r8) then
   ! no prior spread, obs has no effect
   return
end if

select case (filter_kind)

case (KIND_EAKF)
   call obs_increment_eakf(ens, ens_size, prior_mean, prior_var, obs, obs_var, obs_inc, a)

case (KIND_ENKF)
   call obs_increment_enkf(ens, ens_size, prior_var, obs, obs_var, obs_inc)

case (KIND_KERNEL)
   call obs_increment_kernel(ens, ens_size, obs, obs_var, obs_inc)

case (KIND_PARTICLE)
   call obs_increment_particle(ens, ens_size, obs, obs_var, obs_inc)

case (KIND_RHF)
   call obs_increment_rank_histogram(ens, ens_size, prior_var, obs, obs_var, obs_inc)

case (KIND_GAMMA)
   call obs_increment_gamma(ens, ens_size, prior_mean, prior_var, obs, obs_var, obs_inc)

case (KIND_BNRHF)
   ! bounded normal likelihood, as DART's dispatcher builds it before calling the kernel
   do i = 1, ens_size
      likelihood(i) = get_truncated_normal_like(ens(i), obs, obs_var, bb, ba, &
                                                lower_bound, upper_bound)
   end do
   like_sum = sum(likelihood)
   if (like_sum <= 0.0_r8) then
      ! flat likelihood after underflow: no increments
      dart_obs_increment = 3
      return
   end if
   likelihood = likelihood / like_sum
   call obs_increment_bounded_norm_rhf(ens, likelihood, ens_size, prior_var, obs_inc, &
                                       bb, ba, lower_bound, upper_bound)

case (KIND_KDE)
   call obs_increment_kde(ens, ens_size, obs, obs_var, bb, ba, lower_bound, upper_bound, obs_inc)

case default
   dart_obs_increment = 2

end select

end function dart_obs_increment


!> Regress one obs increment onto a batch of state elements.
!>
!> DART's update_from_obs_inc() handles a single scalar state element; looping here
!> in fortran rather than in python keeps this to one ctypes call per observation,
!> matching the granularity of NEDAS's own update_ensemble().
!>
!> state is (num_state, ens_size) in fortran order, which is the same memory layout
!> as a C-ordered numpy array of shape (ens_size, num_state) -- so NEDAS's
!> (nens, nfld*nloc) state block is passed through without a copy.
subroutine dart_update_from_obs_inc(ens_size, num_state, obs_prior, obs_inc, net_a, &
                                    state, lfactor) bind(c, name='dart_update_from_obs_inc')

integer(c_int), intent(in), value :: ens_size, num_state
real(c_double), intent(in)        :: obs_prior(ens_size), obs_inc(ens_size)
real(c_double), intent(in), value :: net_a
real(c_double), intent(inout)     :: state(num_state, ens_size)
real(c_double), intent(in)        :: lfactor(num_state)

integer  :: j
real(r8) :: obs_prior_mean, obs_prior_var, ens(ens_size), state_inc(ens_size), reg_coef

obs_prior_mean = sum(obs_prior) / ens_size
obs_prior_var  = sum((obs_prior - obs_prior_mean)**2) / (ens_size - 1)

! no prior spread in obs space: no regression is defined, leave the state alone
if (obs_prior_var <= 0.0_r8) return

do j = 1, num_state
   if (lfactor(j) <= 0.0_r8) cycle
   ens = state(j, :)
   call update_from_obs_inc(obs_prior, obs_prior_mean, obs_prior_var, obs_inc, &
                            ens, ens_size, state_inc, reg_coef, net_a)
   state(j, :) = ens + lfactor(j) * state_inc
end do

end subroutine dart_update_from_obs_inc

end module dart_kernels_mod
