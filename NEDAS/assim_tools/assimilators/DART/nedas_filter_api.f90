! C entry point: run DART's filter_main on NEDAS's communicator. MPI stays up for NEDAS.
module nedas_filter_api

use, intrinsic :: iso_c_binding, only : c_int
use mpi_utilities_mod, only : initialize_mpi_utilities
use filter_mod,        only : filter_main

implicit none
private

logical :: mpi_up = .false.

contains

subroutine dart_filter_run(fcomm) bind(c, name='dart_filter_run')
integer(c_int), value :: fcomm
if (.not. mpi_up) then
   call initialize_mpi_utilities('filter', communicator=int(fcomm))
   mpi_up = .true.
endif
call filter_main()
end subroutine dart_filter_run

end module nedas_filter_api
