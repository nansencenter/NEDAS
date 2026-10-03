! Advance one bgrid_solo state (DART netCDF restart) by a given interval.
!
! DART's own integrate_model cannot be used standalone: its target time comes from a
! model-advance file written by DART's async machinery. This driver is the same program
! with the target time taken from &nedas_bgrid_advance_nml in input.nml:
!   advance_days, advance_seconds : interval to advance from the time stored in ic_file
! The state is read from ic_file and the advanced state written to ud_file.
!
! With cold_start = .true. no state is read: the model's own cold start (bgrid_cold_start,
! a state at rest with the Held-Suarez equilibrium temperature structure) is written to
! ud_file at time init_days/init_seconds. ic_file still has to exist, as a template with the
! right dimensions (model_nml template_file), and is only used to set up the state layout.
program nedas_bgrid_advance

use time_manager_mod,     only : time_type, operator(<), operator(+), set_time, print_time
use utilities_mod,        only : error_handler, E_MSG, nmlfileunit, do_nml_file, do_nml_term, &
                                 find_namelist_in_file, check_namelist_read
use assim_model_mod,      only : static_init_assim_model, get_model_size
use model_mod,            only : init_conditions
use obs_model_mod,        only : advance_state
use ensemble_manager_mod, only : init_ensemble_manager, ensemble_type, &
                                 all_copies_to_all_vars, all_vars_to_all_copies, set_current_time, &
                                 allocate_vars
use mpi_utilities_mod,    only : initialize_mpi_utilities, finalize_mpi_utilities
use state_vector_io_mod,  only : read_state, write_state
use io_filenames_mod,     only : file_info_type, io_filenames_init, set_io_copy_flag, &
                                 set_file_metadata, READ_COPY, WRITE_COPY
use types_mod,            only : i8

implicit none

character(len=*), parameter :: source = 'nedas_bgrid_advance.f90'

type(ensemble_type)  :: ens_handle
type(time_type)      :: model_time, target_time
type(file_info_type) :: input_file_info, output_file_info
integer              :: iunit, rc
integer(i8)          :: model_size

character(len=256) :: ic_file = 'temp_ic.nc'
character(len=256) :: ud_file = 'temp_ud.nc'
integer            :: advance_days = 0, advance_seconds = 0
logical            :: cold_start = .false.
integer            :: init_days = 0, init_seconds = 0

namelist /nedas_bgrid_advance_nml/ ic_file, ud_file, advance_days, advance_seconds, &
                                cold_start, init_days, init_seconds

call initialize_mpi_utilities('nedas_bgrid_advance')

call find_namelist_in_file('input.nml', 'nedas_bgrid_advance_nml', iunit)
read(iunit, nml=nedas_bgrid_advance_nml, iostat=rc)
call check_namelist_read(iunit, rc, 'nedas_bgrid_advance_nml')

call static_init_assim_model()
model_size = get_model_size()
call init_ensemble_manager(ens_handle, num_copies=1, num_vars=model_size, transpose_type_in=2)

call io_filenames_init(output_file_info, 1, cycling=.true., single_file=.true.)
call set_file_metadata(output_file_info, 1, (/ud_file/), 'temp_ud', 'advanced member')
call set_io_copy_flag(output_file_info, 1, 1, WRITE_COPY, num_output_ens=1)

if (cold_start) then
   call allocate_vars(ens_handle)
   call init_conditions(ens_handle%vars(:, 1))
   model_time = set_time(init_seconds, init_days)
   ens_handle%time(1) = model_time
   call all_vars_to_all_copies(ens_handle)
   call set_current_time(ens_handle, model_time)
else
   call io_filenames_init(input_file_info, 1, cycling=.true., single_file=.true.)
   call set_file_metadata(input_file_info, 1, (/ic_file/), 'temp_ic', 'initial condition')
   call set_io_copy_flag(input_file_info, 1, READ_COPY)
   call read_state(ens_handle, input_file_info, .true., model_time)

   target_time = model_time + set_time(advance_seconds, advance_days)

   ! adv_1step works on whole state vectors (vars storage); the file I/O on copies storage
   call all_copies_to_all_vars(ens_handle)
   if (ens_handle%time(1) < target_time) &
      call advance_state(ens_handle, 1, target_time, 0, '', 1, output_file_info, input_file_info)
   call all_vars_to_all_copies(ens_handle)
   call set_current_time(ens_handle, ens_handle%time(1))   ! the time write_state stamps the file with
endif

call write_state(ens_handle, output_file_info)
call finalize_mpi_utilities()

end program nedas_bgrid_advance
