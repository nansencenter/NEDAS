module dart_wrapper

use types_mod, only : r8, missing_r8

use distribution_params_mod, only : distribution_params_type, deallocate_distribution_params, &
                                    NORMAL_DISTRIBUTION, BOUNDED_NORMAL_RH_DISTRIBUTION, &
                                    GAMMA_DISTRIBUTION, BETA_DISTRIBUTION,               &
                                    LOG_NORMAL_DISTRIBUTION, UNIFORM_DISTRIBUTION,       &
                                    PARTICLE_FILTER_DISTRIBUTION, KDE_DISTRIBUTION

use probit_transform_mod, only: transform_to_probit

implicit none
contains

  subroutine py_transform_to_probit(ens_size, state_ens_in, distribution_type, probit_ens, &
                                    use_input_p, bounded_below, bounded_above, &
                                    lower_bound, upper_bound, ierr)
    integer, intent(in) :: ens_size
    real(r8), intent(in) :: state_ens_in(ens_size)
    integer, intent(in) :: distribution_type
    real(r8), intent(out) :: probit_ens(ens_size)
    logical, intent(in) :: use_input_p
    logical, intent(in) :: bounded_below, bounded_above
    real(r8), intent(in) :: lower_bound, upper_bound
    integer, intent(out) :: ierr

    type(distribution_params_type) :: p

    ! You may need to initialize p here if required
    call transform_to_probit(ens_size, state_ens_in, distribution_type, p, probit_ens, &
                             use_input_p, bounded_below, bounded_above, lower_bound, upper_bound, ierr)
  end subroutine py_transform_to_probit

end module dart_wrapper

