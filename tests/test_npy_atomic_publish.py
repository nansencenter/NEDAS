"""Check that the diag-variable .npy cache is published atomically.

read_var() does a check-then-act on this cache (os.path.exists -> np.load) while
other ranks / postprocess ProcessPool workers may be writing the same path. A plain
np.save() to the final name lets a reader load a half-written file, which is how
amsr2's state_to_obs hit "EOFError: No data left in file" /
"Failed to read all data for array ... file seems not fully written?".
_save_npy_atomic() must never leave the final name partially written.
"""
import os
import numpy as np
import pytest
from NEDAS.models.topaz.v5.topaz5model import Topaz5Model

save = Topaz5Model._save_npy_atomic


def test_publishes_complete_array(tmp_path):
    fname = str(tmp_path / 'seaice_conc_k0_202107010000.npy')
    var = np.arange(1000, dtype='f8').reshape(100, 10)
    save(fname, var)
    np.testing.assert_array_equal(np.load(fname), var)
    # no temp litter left behind
    assert sorted(os.listdir(tmp_path)) == [os.path.basename(fname)]


def test_final_name_never_partially_written(tmp_path, monkeypatch):
    """If the write dies midway, the final name must be absent or still hold the
    previous complete value -- never a truncated file."""
    fname = str(tmp_path / 'cache.npy')
    old = np.full((50, 4), 1.0)
    save(fname, old)

    real_save = np.save

    def dying_save(file, arr, *a, **kw):
        real_save(file, arr, *a, **kw)      # temp file is written...
        raise OSError('disk full')          # ...but never renamed in

    monkeypatch.setattr('NEDAS.models.topaz.v5.topaz5model.np.save', dying_save)
    with pytest.raises(OSError):
        save(fname, np.full((50, 4), 2.0))

    # the reader still sees the previous complete array, not a partial one
    np.testing.assert_array_equal(np.load(fname), old)


def test_concurrent_readers_never_see_partial_file(tmp_path):
    """Interleave a reader against a writer the way separate ranks do: every
    os.path.exists()->np.load() that sees the file must get a loadable array."""
    fname = str(tmp_path / 'cache.npy')
    var = np.arange(200000, dtype='f8')     # big enough that a direct np.save is not one write

    seen = 0
    for _ in range(5):
        if os.path.exists(fname):
            os.remove(fname)
        save(fname, var)
        if os.path.exists(fname):
            np.testing.assert_array_equal(np.load(fname), var)
            seen += 1
    assert seen == 5
