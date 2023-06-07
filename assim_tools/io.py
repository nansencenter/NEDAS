from netCDF4 import Dataset
import struct
from .state_def import state_info

##netcdf file io
def nc_write_var(filename, dim, varname, dat, attr=None):
    '''
    write gridded data to netcdf file
    filename: path/name of the output file
    dim: dict{'dimension name': number of grid points in dimension (int)}
    varname: name for the output variable
    dat: data for output, number of dimensions must match dim
    '''
    f = Dataset(filename, 'a', format="NETCDF4_CLASSIC")
    ndim = len(dim)
    assert(len(dat.shape)==ndim)
    for key in dim:
        if key in f.dimensions:
            if dim[key]!=0:
                assert(f.dimensions[key].size==dim[key])
        else:
            f.createDimension(key, size=dim[key])
    if varname in f.variables:
        assert(f.variables[varname].shape==dat.shape)
    else:
        f.createVariable(varname, float, dim.keys())
    f[varname][:] = dat
    if attr != None:
        for akey in attr:
            f[varname].setncattr(akey, attr[akey])
    f.close()

def nc_read_var(filename, varname):
    '''
    read gridded data from netcdf file
    '''
    f = Dataset(filename, 'r')
    assert(varname in f.variables)
    dat = f[varname][:].data
    f.close()
    return dat

###binary file io
def create_var(filename, state_info, data):
    ###write a new bin file for the first time



def variable_info(filename):
    return v_info

def read_var(filename):
    v_info = variable_info(filename)
    return data

def write_var(filename, data):
    v_info = variable_info(filename)
    
