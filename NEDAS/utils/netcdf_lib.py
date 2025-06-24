from typing import Optional, Dict, Literal
import numpy as np
from netCDF4 import Dataset
from NEDAS.utils.parallel import Comm

AccessMode = Literal['r', 'w', 'a', 'r+']

def nc_open(filename: str, mode: AccessMode, comm: Optional[Comm]=None) -> Dataset:
    """
    Open a netCDF file.

    Args:
        filename (str): Path to the netCDF file.
        mode (str): Open mode, (e.g. :code:`'r'`, :code:`'w'`).
        comm (Comm, optional): MPI communicator object. If None, open the netCDF4.Dataset normally.
            If communicator is available, try :code:`parallel=True` open when opening the file.
            If that's not supported, acquire a file lock in communicator for blocking serial access of the file.

    Returns:
        netCDF4.Dataset: netCDF file handle.
    """
    if comm is None:
        return Dataset(filename, mode, format='NETCDF4')
    else:
        if comm.parallel_io:
            return Dataset(filename, mode, format='NETCDF4', parallel=True)
        else:
            comm.acquire_file_lock(filename)
            try:
                return Dataset(filename, mode, format='NETCDF4')
            except Exception:
                comm.release_file_lock(filename)
                raise

def nc_close(filename: str, f: Dataset, comm: Optional[Comm]=None) -> None:
    """
    Close the netCDF file handle.

    Args:
        filename (str): Path to the opened netCDF file.
        f (netCDF4.Dataset): netCDF4 Dataset file handle.
        comm (Comm, optional): MPI communicator object. If None, just close the file directly.
            If communicator is specified, release the file lock after closing it.
    """
    f.close()
    if comm is not None and not comm.parallel_io:
        comm.release_file_lock(filename)

def nc_write_var(filename: str,
                 dim: Dict[str,Optional[int]],
                 varname: str,
                 dat: np.ndarray,
                 dtype: Optional[str]=None,
                 recno: Optional[Dict[str,int]]=None,
                 attr: Optional[Dict]=None,
                 comm: Optional[Comm]=None) -> None:
    """
    Write a variable to a netCDF file.

    Args:
        filename (str): Path to the output netCDF file.
        dim (dict): Dictionary {dimension name (str): dimension size (int)} of each dimension.
            The dimension size can be None if it is `unlimited` dimension (can append more records afterwards)
        varname (str): Name of the output variable. Variable groups are supported, use :code:`'group/varname'` as varname.
        dat (np.ndarray): Data for output, number of its dimensions must match :code:`dim` (excluding unlimited dimensions).
        dtype (str, optional): Data type to convert the input data to.
        recno (dict, optional): Dictionary {dimension name (str): record number (int)}, indicating the index in unlimited dimensions
            for the data to be written to. Each unlimited dimension defined in :code:`dim` should have
            a corresponding :code:`recno` entry.
        attr (dict, optional): Dictionary {name of attribute (str): value (str)}, additional attributes to be added.
        comm (Comm, optional): MPI communicator object, handling parallel I/O and make sure thread-safe writing of data.
    """
    f = nc_open(filename, 'a', comm)

    varname_parts = varname.split('/')
    group_path, varname = '/'.join(varname_parts[:-1]), varname_parts[-1]

    group = f
    if group_path:
        for part in group_path.split('/'):
            if part not in group.groups:
                group.createGroup(part)
            group = group.groups[part]

    if dtype is None:
        if isinstance(dat, np.ndarray):
            dtype = dat.dtype
        else:
            dtype = type(dat)

    ndim = len(dim)
    s = ()  ##slice for each dimension
    d = 0
    for i, name in enumerate(dim):
        if dim[name] is None:
            assert(name in recno), "unlimited dimension "+name+" doesn't have a recno"
            s += (recno[name],)
        else:
            s += (slice(None),)
            assert(dat.shape[d] == dim[name]), "dat size for dimension "+name+" mismatch with dim["+name+"]={}".format(dim[name])
            d += 1

        if name in group.dimensions:
            if dim[name] is not None:
                assert(group.dimensions[name].size==dim[name]), "dimension "+name+" size ({}) mismatch with file ({})".format(dim[name], group.dimensions[name].size)
            else:
                if not group.dimensions[name].isunlimited():
                    assert(recno[name] < group.dimensions[name].size), "recno for dimension "+name+" exceeds file size"
        else:
            group.createDimension(name, size=dim[name])

    if varname not in group.variables:
        group.createVariable(varname, dtype, tuple(dim.keys())) # type: ignore

    if isinstance(dat, np.ndarray):
        dat = dat.astype(dtype)
    else:
        dat = dtype(dat)

    group[varname][s] = dat  ##write dat to file

    if attr is not None:
        for akey in attr:
            group[varname].setncattr(akey, attr[akey])

    nc_close(filename, f, comm)

def nc_read_var(filename: str, varname: str, comm: Optional[Comm]=None) -> np.ndarray:
    """
    Read a variable from a netCDF file.

    This function by default reads the entire variable, if you only want a slice, it is more efficient to use
    netCDF4.Dataset handle directly.

    Args:
        filename (str): Path to the netCDF file for reading.
        varname (str): Name of the variable to read.
        comm (Comm, optional): MPI communicator object.

    Returns:
        np.ndarray: Variable read from the file.
    """
    f = nc_open(filename, 'r', comm)

    varname_parts = varname.split('/')
    group_path, varname = '/'.join(varname_parts[:-1]), varname_parts[-1]

    group = f
    if group_path:
        for part in group_path.split('/'):
            if part not in group.groups:
                group.createGroup(part)
            group = group.groups[part]

    assert varname in group.variables, f"variable '{varname}' is not defined in {filename}"

    dat = group[varname][...]
    dat_out = dat.data
    dat_out[dat.mask] = np.nan

    nc_close(filename, f, comm)

    return dat_out
