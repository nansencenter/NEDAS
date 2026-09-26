"""Regression check for the .b lost-update hazard behind topaz5's fixhycom failure.

ABFileRestart.overwrite_field() writes its field to one .a record (a distinct byte
offset, so concurrent writers never collide there) but rewrites the WHOLE .b from the
min/max snapshot taken when that handle opened the file. Two handles open at once
therefore lose each other's .b updates, leaving .b advertising pre-analysis min/max
for records whose .a really did change -- which is exactly what makes the external
fixhycom binary abort with "Inconsistency between .a and .b files (m_get_mod_fld)".

topaz5model.write_var() serializes this read-modify-write on the per-file lock, so the
interleaved pattern below must never happen there. This test pins the underlying
semantics: if overwrite_field is ever changed to update .b in place (per-record), the
`interleaved` case starts passing and the lock is no longer load-bearing.
"""
import numpy as np
import pytest
from NEDAS.models.topaz.abfile import ABFileRestart

IDM, JDM = 16, 8


def _make_restart(basename, fields):
    """Write a minimal 2-field restart pair; fields is [(name, k, array), ...]."""
    f = ABFileRestart(basename, 'w', idm=IDM, jdm=JDM, mask=True)
    f.write_header(21, 22, 3, 1, 12675456, 44012.0, 25.0)
    for name, k, arr in fields:
        f.write_field(arr, np.isnan(arr), name, k, 1)
    f.close()


def _bminmax(basename, name, k):
    f = ABFileRestart(basename, 'r', idm=IDM, jdm=JDM, mask=True)
    try:
        return f.bminmax(name, k)
    finally:
        f.close()


@pytest.fixture
def restart(tmp_path):
    base = str(tmp_path / 'restart.2021_182_00_0000')
    a = np.full((JDM, IDM), 5.0); a[0, 0] = 0.0     # like dp: min is an exact 0.0
    b = np.full((JDM, IDM), 7.0); b[0, 0] = 0.0
    _make_restart(base, [('dp', 1, a), ('dp', 2, b)])
    assert _bminmax(base, 'dp', 1) == (0.0, 5.0)
    assert _bminmax(base, 'dp', 2) == (0.0, 7.0)
    return base


def _new_field(val):
    """A field whose old exact-zero point is now masked -- so its .b min MUST change."""
    fld = np.full((JDM, IDM), val)
    mask = np.zeros((JDM, IDM), dtype=bool); mask[0, 0] = True
    return fld, mask


def test_serialized_writes_keep_both_b_entries(restart):
    """One handle at a time (what the file lock guarantees): both .b entries survive."""
    for k, val in ((1, 5.0), (2, 7.0)):
        fld, mask = _new_field(val)
        f = ABFileRestart(restart, 'r+', idm=IDM, jdm=JDM, mask=True)
        f.overwrite_field(fld, mask, 'dp', level=k, tlevel=1)
        f.close()

    assert _bminmax(restart, 'dp', 1) == (5.0, 5.0)
    assert _bminmax(restart, 'dp', 2) == (7.0, 7.0)


def test_interleaved_writes_lose_a_b_entry(restart):
    """Two concurrent handles (no lock): the .a records are both correct but the
    second writer's whole-file .b rewrite reverts the first writer's entry to its
    stale pre-analysis min -- the fixhycom "Inconsistency" failure, reproduced."""
    f1 = ABFileRestart(restart, 'r+', idm=IDM, jdm=JDM, mask=True)
    f2 = ABFileRestart(restart, 'r+', idm=IDM, jdm=JDM, mask=True)   # snapshot taken now
    fld1, mask1 = _new_field(5.0)
    fld2, mask2 = _new_field(7.0)
    f1.overwrite_field(fld1, mask1, 'dp', level=1, tlevel=1)
    f2.overwrite_field(fld2, mask2, 'dp', level=2, tlevel=1)         # rewrites all of .b
    f1.close(); f2.close()

    assert _bminmax(restart, 'dp', 2) == (7.0, 7.0)     # last writer wins
    assert _bminmax(restart, 'dp', 1) == (0.0, 5.0), \
        "expected the stale pre-analysis min to survive on record 1 (lost update)"

    # the .a record itself is correct -- the masked point really was written as
    # spval, so nothing in the payload is zero any more. Only .b is stale.
    with open(restart + '.a', 'rb') as fh:
        raw = np.frombuffer(fh.read(), dtype='>f4')[:IDM * JDM].astype('f8')
    assert not (raw[raw < 2.0 ** 99] == 0.0).any()

    # so reading record 1 back trips the same check fixhycom's m_get_mod_fld does
    f = ABFileRestart(restart, 'r', idm=IDM, jdm=JDM, mask=True)
    try:
        with pytest.raises(ValueError, match='values inconsistent'):
            f.read_field('dp', level=1, tlevel=1)
    finally:
        f.close()
