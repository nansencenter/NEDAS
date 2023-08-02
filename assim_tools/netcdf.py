import numpy as np
from netCDF4 import Dataset

##netcdf file io
def nc_write_var(filename, dim, varname, dat, recno=None, attr=None):
    ###write gridded data to netcdf file
    ###filename: path/name of the output file
    ###dim: dict{'dimension name': number of grid points in dimension (int); None for unlimited}
    ###varname: name for the output variable
    ###dat: data for output, number of dimensions must match dim (excluding unlimited dims)
    ###recno: dict{'dimension name': record_number, ...}, each unlimited dim should have a recno entry
    ###attr: attribute dict{'name of attr':'value of attr'}
    f = Dataset(filename, 'a', format='NETCDF4')
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
        f.createVariable(varname, np.float32, dim.keys())

    f[varname][s] = dat  ##write dat to file
    if attr is not None:
        for akey in attr:
            f[varname].setncattr(akey, attr[akey])
    f.close()

def nc_read_var(filename, varname):
    ###read gridded data from netcdf file
    f = Dataset(filename, 'r')
    assert(varname in f.variables)
    dat = f[varname][...].data
    f.close()
    return dat

