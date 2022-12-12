import numpy as np

def write_grid_nc(filename, dim, varname, dat):
    '''
    write gridded data to netcdf file
    filename: path/name of the output file
    dim: dict{'dimension name': number of grid points in dimension (int)}
    varname: name for the output variable
    dat: data for output, number of dimensions must match dim
    '''
    from netCDF4 import Dataset
    f = Dataset(filename, 'a', format="NETCDF4")
    ndim = len(dim)
    assert(len(dat.shape)==ndim)
    for key in dim:
        if key in f.dimensions:
            assert(f.dimensions[key].size==dim[key])
        else:
            f.createDimension(key, size=dim[key])
    if varname in f.variables:
        assert(f.variables[varname].shape==dat.shape)
    else:
        f.createVariable(varname, float, dim.keys())
    f[varname][:] = dat
    return f

def read_grid_nc(filename, varname):
    from netCDF4 import Dataset
    f = Dataset(filename, 'r')
    assert(varname in f.variables)
    dat = f[varname][:]
    return dat


# def read


# def write_grid_ab():
