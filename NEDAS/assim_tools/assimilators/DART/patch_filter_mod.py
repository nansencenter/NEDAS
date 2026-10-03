"""
Write a copy of DART's filter_mod.f90 whose file I/O goes to nedas_hooks_mod instead.
Each patch must match exactly the expected number of times, else this fails.

Usage: python patch_filter_mod.py DART/.../filter_mod.f90 OUT.f90
"""
import re
import sys

PATCHES = [
    # (pattern, replacement, expected count)
    (r'call read_obs_seq_header\(obs_sequence_in_name,[^)]*\)',
     'call nedas_obs_seq_header(tnum_copies, tnum_qc, tnum_obs, tmax_num_obs)', 1),
    (r'call read_obs_seq\(obs_sequence_in_name, copies_num_inc, qc_num_inc, 0, seq\)',
     'call nedas_read_obs_seq(copies_num_inc, qc_num_inc, seq)', 1),
    (r'call initialize_file_information\([^)]*\)',
     '! NEDAS: no state files', 1),
    (r'call check_file_info_variable_shape\(file_info_output, state_ens_handle\)',
     '! NEDAS: no state files', 1),
    (r'call read_state\(state_ens_handle, file_info_input,[^)]*\)',
     'call nedas_read_state(state_ens_handle, read_time_from_file, time1, prior_inflate, post_inflate)', 1),
    (r'call get_obs_ens_distrib_state\((?=[^)]*isprior=\.false\.)',
     'call nedas_obs_ens_distrib_state(', 1),
    (r'call write_state\(state_ens_handle, file_info_output\)',
     'call nedas_write_state(state_ens_handle, prior_inflate, post_inflate)', 1),
    (r'call write_obs_seq\(seq, obs_sequence_out_name\)',
     'call nedas_write_obs_seq(seq, obs_sequence_out_name)', 2),
    (r'(use distribution_params_mod, only : distribution_params_type\n)',
     r'\1use nedas_hooks_mod, only : nedas_read_state, nedas_write_state, nedas_obs_seq_header, &\n'
     r'                            nedas_read_obs_seq, nedas_obs_ens_distrib_state, nedas_write_obs_seq\n', 1),
]


def patch(src: str) -> str:
    for pattern, repl, count in PATCHES:
        src, n = re.subn(pattern, repl, src)
        if n != count:
            raise RuntimeError(f"patch matched {n} times, expected {count}: {pattern}")
    return src


if __name__ == '__main__':
    src_file, out_file = sys.argv[1:3]
    with open(src_file) as f:
        src = f.read()
    with open(out_file, 'w') as f:
        f.write(patch(src))
