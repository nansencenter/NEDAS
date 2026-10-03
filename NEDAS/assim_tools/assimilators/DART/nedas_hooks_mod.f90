! In-memory replacements for filter_main's file I/O, and the C API NEDAS calls.
!
! patch_filter_mod.py swaps the calls in a copy of DART's filter_mod.f90 for these.
! Arrays stay owned by NEDAS (numpy); only pointers are kept here.
module nedas_hooks_mod

use, intrinsic :: iso_c_binding
use types_mod,            only : r8, MISSING_R8
use time_manager_mod,     only : time_type, set_time
use utilities_mod,        only : error_handler, E_ERR
use mpi_utilities_mod,    only : my_task_id, task_count
use ensemble_manager_mod, only : ensemble_type, compute_copy_mean_var
use adaptive_inflate_mod, only : adaptive_inflate_type, do_ss_inflate, mean_from_restart, &
                                 sd_from_restart, get_inflate_mean, get_inflate_sd, &
                                 get_inflation_mean_copy, get_inflation_sd_copy
use obs_sequence_mod,     only : obs_sequence_type, obs_type, init_obs_sequence, init_obs, &
                                 set_copy_meta_data, set_qc_meta_data, set_obs_def, &
                                 set_obs_values, set_qc, append_obs_to_seq, get_obs_from_key, &
                                 get_qc, destroy_obs, write_obs_seq
use obs_def_mod,          only : obs_def_type, set_obs_def_location, set_obs_def_type_of_obs, &
                                 set_obs_def_time, set_obs_def_error_variance, set_obs_def_key, &
                                 set_obs_def_external_FO
use obs_kind_mod,         only : get_index_for_type_of_obs
use location_mod,         only : set_location
use quality_control_mod,  only : input_qc_ok, get_dart_qc
use model_mod,            only : nedas_set_state_meta, nedas_set_periodic

implicit none
private

public :: nedas_read_state, nedas_write_state, nedas_obs_seq_header, nedas_read_obs_seq, &
          nedas_obs_ens_distrib_state, nedas_write_obs_seq

character(len=*), parameter :: source = 'nedas_hooks_mod.f90'

abstract interface
   subroutine callback() bind(c)
   end subroutine callback
end interface

! state block of this task: state(nloc, nens) is numpy (nens, nloc); inf(nloc, 4)
integer     :: nens = 0, nloc = 0, nmax = 0, t_days = 0, t_secs = 0
type(c_ptr) :: state_p = c_null_ptr, inf_p = c_null_ptr

! obs, replicated on all tasks; prior/post (nobs, nens) are numpy (nens, nobs)
integer     :: nobs = 0
type(c_ptr) :: otype_p, ox_p, oy_p, oz_p, oday_p, osec_p, oval_p, oerr_p, prior_p
type(c_ptr) :: post_p = c_null_ptr
procedure(callback), pointer :: post_callback => null()

logical :: write_final = .false.

contains

!------------------------------------------------------------------ C API

subroutine dart_filter_set_state(nens_in, nloc_in, nmax_in, state, x, y, z, qty, inf, &
                                 days, secs) bind(c, name='dart_filter_set_state')
integer(c_int), value :: nens_in, nloc_in, nmax_in, days, secs
type(c_ptr),    value :: state, inf
real(c_double), intent(in) :: x(nloc_in), y(nloc_in), z(nloc_in)
integer(c_int), intent(in) :: qty(nloc_in)

nens = nens_in; nloc = nloc_in; nmax = nmax_in
state_p = state; inf_p = inf
t_days = days; t_secs = secs
call nedas_set_state_meta(nloc, nmax, x, y, z, qty)
end subroutine dart_filter_set_state


subroutine dart_filter_set_periodic(x_on, xmin, xmax, y_on, ymin, ymax) &
                                    bind(c, name='dart_filter_set_periodic')
integer(c_int), value :: x_on, y_on
real(c_double), value :: xmin, xmax, ymin, ymax
call nedas_set_periodic(x_on /= 0, xmin, xmax, y_on /= 0, ymin, ymax)
end subroutine dart_filter_set_periodic


subroutine dart_filter_set_obs(nobs_in, otype, x, y, z, days, secs, val, errvar, prior) &
                               bind(c, name='dart_filter_set_obs')
integer(c_int), value :: nobs_in
type(c_ptr),    value :: otype, x, y, z, days, secs, val, errvar, prior

nobs = nobs_in
otype_p = otype; ox_p = x; oy_p = y; oz_p = z
oday_p = days; osec_p = secs; oval_p = val; oerr_p = errvar; prior_p = prior
end subroutine dart_filter_set_obs


subroutine dart_filter_set_posterior(post, cb) bind(c, name='dart_filter_set_posterior')
type(c_ptr),    value :: post
type(c_funptr), value :: cb
post_p = post
call c_f_procpointer(cb, post_callback)
end subroutine dart_filter_set_posterior


subroutine dart_filter_set_write_obs_seq(flag) bind(c, name='dart_filter_set_write_obs_seq')
integer(c_int), value :: flag
write_final = (flag /= 0)
end subroutine dart_filter_set_write_obs_seq




!------------------------------------------------------------------ state hooks

subroutine check_layout(ens_handle)
type(ensemble_type), intent(in) :: ens_handle
integer :: k
if (ens_handle%num_copies - ens_handle%num_extras /= nens) &
   call error_handler(E_ERR, 'nedas hooks', 'ens_size differs from NEDAS nens', source)
if (ens_handle%my_num_vars /= nmax) &
   call error_handler(E_ERR, 'nedas hooks', 'unexpected number of local state entries', source)
do k = 1, nmax
   if (ens_handle%my_vars(k) /= int(k-1, 8)*task_count() + my_task_id() + 1) &
      call error_handler(E_ERR, 'nedas hooks', 'state ownership differs from NEDAS blocks', source)
end do
end subroutine check_layout


subroutine nedas_read_state(ens_handle, read_time_from_file, model_time, prior_inflate, post_inflate)
type(ensemble_type),         intent(inout) :: ens_handle
logical,                     intent(in)    :: read_time_from_file
type(time_type),             intent(inout) :: model_time
type(adaptive_inflate_type), intent(in)    :: prior_inflate, post_inflate

real(r8), pointer :: st(:,:), inf(:,:)

call check_layout(ens_handle)
call c_f_pointer(state_p, st, [nloc, nens])
call c_f_pointer(inf_p, inf, [nloc, 4])

ens_handle%copies(1:nens, 1:nloc) = transpose(st)
ens_handle%copies(1:nens, nloc+1:nmax) = 0.0_r8

if (read_time_from_file) model_time = set_time(t_secs, t_days)
ens_handle%time(:) = model_time

call fill_inflation(ens_handle, prior_inflate, inf(:, 1), inf(:, 2))
call fill_inflation(ens_handle, post_inflate,  inf(:, 3), inf(:, 4))
end subroutine nedas_read_state


! same rules as DART's fill_inf_from_namelist_value, with restart values from NEDAS
subroutine fill_inflation(ens_handle, inflate, mean_in, sd_in)
type(ensemble_type),         intent(inout) :: ens_handle
type(adaptive_inflate_type), intent(in)    :: inflate
real(r8),                    intent(in)    :: mean_in(:), sd_in(:)
integer :: mc, sc

mc = get_inflation_mean_copy(inflate)
sc = get_inflation_sd_copy(inflate)

if (.not. do_ss_inflate(inflate)) then
   ens_handle%copies(mc, :) = 1.0_r8
   ens_handle%copies(sc, :) = 0.0_r8
   return
endif

ens_handle%copies(mc, :) = get_inflate_mean(inflate)
if (mean_from_restart(inflate)) ens_handle%copies(mc, 1:nloc) = mean_in
ens_handle%copies(sc, :) = get_inflate_sd(inflate)
if (sd_from_restart(inflate)) ens_handle%copies(sc, 1:nloc) = sd_in
end subroutine fill_inflation


subroutine nedas_write_state(ens_handle, prior_inflate, post_inflate)
type(ensemble_type),         intent(in) :: ens_handle
type(adaptive_inflate_type), intent(in) :: prior_inflate, post_inflate

real(r8), pointer :: inf(:,:)

call copy_state_out(ens_handle)
call c_f_pointer(inf_p, inf, [nloc, 4])
inf(:, 1) = ens_handle%copies(get_inflation_mean_copy(prior_inflate), 1:nloc)
inf(:, 2) = ens_handle%copies(get_inflation_sd_copy(prior_inflate),   1:nloc)
inf(:, 3) = ens_handle%copies(get_inflation_mean_copy(post_inflate),  1:nloc)
inf(:, 4) = ens_handle%copies(get_inflation_sd_copy(post_inflate),    1:nloc)
end subroutine nedas_write_state


subroutine copy_state_out(ens_handle)
type(ensemble_type), intent(in) :: ens_handle
real(r8), pointer :: st(:,:)
call c_f_pointer(state_p, st, [nloc, nens])
st = transpose(ens_handle%copies(1:nens, 1:nloc))
end subroutine copy_state_out


!------------------------------------------------------------------ obs hooks

subroutine nedas_obs_seq_header(num_copies, num_qc, num_obs, max_num_obs)
integer, intent(out) :: num_copies, num_qc, num_obs, max_num_obs
num_copies = 1
num_qc = 1
num_obs = nobs
max_num_obs = nobs
end subroutine nedas_obs_seq_header


subroutine nedas_read_obs_seq(copies_inc, qc_inc, seq)
integer,                 intent(in)  :: copies_inc, qc_inc
type(obs_sequence_type), intent(out) :: seq

integer(c_int), pointer :: otype(:), oday(:), osec(:)
real(r8),       pointer :: ox(:), oy(:), oz(:), oval(:), oerr(:), prior(:,:)
type(obs_type)     :: obs
type(obs_def_type) :: def
real(r8), allocatable :: vals(:), qcs(:)
character(len=32) :: tname
integer :: i, itype, ncopy, nqc

call c_f_pointer(otype_p, otype, [nobs])
call c_f_pointer(oday_p, oday, [nobs])
call c_f_pointer(osec_p, osec, [nobs])
call c_f_pointer(ox_p, ox, [nobs])
call c_f_pointer(oy_p, oy, [nobs])
call c_f_pointer(oz_p, oz, [nobs])
call c_f_pointer(oval_p, oval, [nobs])
call c_f_pointer(oerr_p, oerr, [nobs])
call c_f_pointer(prior_p, prior, [nobs, nens])

ncopy = 1 + copies_inc
nqc   = 1 + qc_inc
call init_obs_sequence(seq, ncopy, nqc, max(nobs, 1))
call set_copy_meta_data(seq, 1, 'NEDAS observation')
call set_qc_meta_data(seq, 1, 'NEDAS QC')

call init_obs(obs, ncopy, nqc)
allocate(vals(ncopy), qcs(nqc))
vals = MISSING_R8
qcs  = 0.0_r8

do i = 1, nobs
   write(tname, '(A,I2.2)') 'NEDAS_', otype(i)
   itype = get_index_for_type_of_obs(tname)
   if (itype < 0) call error_handler(E_ERR, 'nedas_read_obs_seq', 'unknown obs type '//trim(tname), source)
   call set_obs_def_location(def, set_location(ox(i), oy(i), oz(i)))
   call set_obs_def_type_of_obs(def, itype)
   call set_obs_def_time(def, set_time(osec(i), oday(i)))
   call set_obs_def_error_variance(def, oerr(i))
   call set_obs_def_key(def, i)
   call set_obs_def_external_FO(def, .true., .false., i, nens, prior(i, :))
   call set_obs_def(obs, def)
   vals(1) = oval(i)
   call set_obs_values(obs, vals)
   call set_qc(obs, qcs)
   call append_obs_to_seq(seq, obs)
end do

call destroy_obs(obs)
end subroutine nedas_read_obs_seq


!> Posterior pass only: H(x_post) comes from NEDAS through post_callback.
!> Mirrors the distributed-state posterior branch of get_obs_ens_distrib_state.
subroutine nedas_obs_ens_distrib_state(ens_handle, obs_fwd_op_ens_handle, &
   qc_ens_handle, seq, keys, obs_val_index, input_qc_index, &
   OBS_ERR_VAR_COPY, OBS_VAL_COPY, OBS_KEY_COPY, &
   OBS_GLOBAL_QC_COPY, OBS_EXTRA_QC_COPY, OBS_MEAN_COPY, &
   OBS_VAR_COPY, isprior, prior_qc_copy)

type(ensemble_type),     intent(inout) :: ens_handle, obs_fwd_op_ens_handle, qc_ens_handle
type(obs_sequence_type), intent(in)    :: seq
integer,                 intent(in)    :: keys(:), obs_val_index, input_qc_index
integer,                 intent(in)    :: OBS_ERR_VAR_COPY, OBS_VAL_COPY, OBS_KEY_COPY
integer,                 intent(in)    :: OBS_GLOBAL_QC_COPY, OBS_EXTRA_QC_COPY
integer,                 intent(in)    :: OBS_MEAN_COPY, OBS_VAR_COPY
logical,                 intent(in)    :: isprior
real(r8),                intent(inout) :: prior_qc_copy(:)

real(r8), pointer :: post(:,:)
type(obs_type) :: obs
real(r8) :: input_qc(1)
integer  :: j, key, qc, istatus(nens), global_qc

if (isprior) call error_handler(E_ERR, 'nedas_obs_ens_distrib_state', 'posterior only', source)
if (.not. associated(post_callback)) call error_handler(E_ERR, &
   'nedas_obs_ens_distrib_state', 'no posterior H(x) callback set by NEDAS', source)

! NEDAS evaluates H on the current posterior (all tasks take part)
call copy_state_out(ens_handle)
call post_callback()
call c_f_pointer(post_p, post, [nobs, nens])

call init_obs(obs, 0, 0)
do j = 1, obs_fwd_op_ens_handle%my_num_vars
   key = keys(obs_fwd_op_ens_handle%my_vars(j))
   call get_obs_from_key(seq, key, obs)
   call get_qc(obs, input_qc, input_qc_index)
   if (.not. input_qc_ok(input_qc(1), global_qc)) cycle

   istatus = 0
   where (post(key, :) /= post(key, :)) istatus = 1   ! NaN: failed forward operator
   where (istatus == 0)
      obs_fwd_op_ens_handle%copies(1:nens, j) = post(key, :)
   elsewhere
      obs_fwd_op_ens_handle%copies(1:nens, j) = MISSING_R8
   end where

   qc = nint(obs_fwd_op_ens_handle%copies(OBS_GLOBAL_QC_COPY, j))
   call get_dart_qc(istatus, nens, .true., .false., .false., qc)
   obs_fwd_op_ens_handle%copies(OBS_GLOBAL_QC_COPY, j) = qc
   qc_ens_handle%copies(:, j) = istatus
end do
call destroy_obs(obs)

call compute_copy_mean_var(obs_fwd_op_ens_handle, 1, nens, OBS_MEAN_COPY, OBS_VAR_COPY)
do j = 1, obs_fwd_op_ens_handle%my_num_vars
   if (any(qc_ens_handle%copies(:, j) /= 0)) then
      obs_fwd_op_ens_handle%copies(OBS_MEAN_COPY, j) = MISSING_R8
      obs_fwd_op_ens_handle%copies(OBS_VAR_COPY,  j) = MISSING_R8
   endif
end do
end subroutine nedas_obs_ens_distrib_state


subroutine nedas_write_obs_seq(seq, fname)
type(obs_sequence_type), intent(in) :: seq
character(len=*),        intent(in) :: fname
if (write_final) call write_obs_seq(seq, fname)
end subroutine nedas_write_obs_seq

end module nedas_hooks_mod
