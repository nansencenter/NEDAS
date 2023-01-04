from netCDF4 import Dataset

def write(filename, dim, varname, dat):
    '''
    write gridded data to netcdf file
    filename: path/name of the output file
    dim: dict{'dimension name': number of grid points in dimension (int)}
    varname: name for the output variable
    dat: data for output, number of dimensions must match dim
    '''
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
    f.close()

def read(filename, varname):
    '''
    read gridded data from netcdf file
    '''
    f = Dataset(filename, 'r')
    assert(varname in f.variables)
    dat = f[varname][:]
    f.close()
    return dat

