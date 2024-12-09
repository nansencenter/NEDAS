import numpy as np
from netCDF4 import Dataset
import time

def nc_open(filename, mode, comm):
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

def nc_close(filename, f, comm):
    f.close()
    if comm is not None and not comm.parallel_io:
        comm.release_file_lock(filename)

def nc_write_var(filename, dim, varname, dat, dtype=None, recno=None, attr=None, comm=None):
    """
    Write a variable to a netcdf file

    Inputs:
    - filename: str
      Path to the output nc file

    - dim: dict(str:int)
      Dimension definition for the variable, 'dimension name':length of dimension (int)
      The dimension length can be None if it is "unlimited" dimension (record dimension)

    - varname: str
      Name of the output variable

    - dat: np.array
      Data for output, number of its dimensions must match dim (excluding unlimited dims)

    - dtype:
      Data type, if not None will convert input dat to dtype

    - recno: dict, optional
      Dictionary {'dimension name': record_number}, each unlimited dimension defined in dim
      should have a corresponding recno entry to note which record to write to.

    - attr: dict, optional
      Additional attribute to output to the variable
      Dictionary {'name of attr':'value of attr'}

    - comm: Comm object
    """
    f = nc_open(filename, 'a', comm)

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

        if name in f.dimensions:
            if dim[name] is not None:
                assert(f.dimensions[name].size==dim[name]), "dimension "+name+" size ({}) mismatch with file ({})".format(dim[name], f.dimensions[name].size)
            else:
                assert(f.dimensions[name].isunlimited())
        else:
            f.createDimension(name, size=dim[name])
    if varname not in f.variables:
        f.createVariable(varname, dtype, dim.keys())

    if isinstance(dat, np.ndarray):
        dat = dat.astype(dtype)
    else:
        dat = dtype(dat)
    f[varname][s] = dat  ##write dat to file
    if attr is not None:
        for akey in attr:
            f[varname].setncattr(akey, attr[akey])

    nc_close(filename, f, comm)

def nc_read_var(filename, varname, comm=None):
    """
    Read a variable from an netcdf file

    This reads the entire variable, if you only want a slice, it is more efficient to use
    netCDF4.Dataset directly.

    Inputs:
    - filename: str
      Path to the netcdf file for reading

    - varname: str
      Name of the variable to read

    - comm: Comm object
    Return:
    - dat: np.array
      The variable read from the file
    """
    f = nc_open(filename, 'r', comm)

    assert varname in f.variables, 'variable '+varname+' is not defined in '+filename
    dat = f[varname][...].data

    nc_close(filename, f, comm)

    return dat

