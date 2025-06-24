"""
"""
from datetime import datetime
import cftime
import netCDF4
import numpy as np

def copy_group(src_group, dst_group, time_index, time_dimension_name, time) -> None:
    """
    Recursively copy a netCDF group, including dimensions, variables, and attributes.
    """
    # Copy global (or group) attributes
    dst_group.setncatts(src_group.__dict__)

    # Copy dimensions
    for name, dimension in src_group.dimensions.items():
        dst_group.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    # Copy variables
    for name, variable in src_group.variables.items():
        # Create the variable in the destination group
        dst_var = dst_group.createVariable(name, variable.datatype, variable.dimensions)
        dst_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})

        if len(variable.dimensions) == 0:
            if name == 'formatted':
                dst_var[0] = time.strftime(dst_var.format)
            if name == 'time':
                dst_var[0] = cftime.date2num(time, dst_var.units)
            continue
        # Copy the data, considering the time slice if relevant
        if time_dimension_name in variable.dimensions:
            # For other time-dependent variables, copy the time slice
            slice_spec = [time_index if dim == time_dimension_name else slice(None) for dim in variable.dimensions]
            dst_var[:] = variable[tuple(slice_spec)]
        else:
            # For time-independent variables, copy all data
            dst_var[:] = variable[:]

    # Recursively copy groups
    for group_name in src_group.groups:
        src_subgroup = src_group.groups[group_name]
        dst_subgroup = dst_group.createGroup(group_name)
        copy_group(src_subgroup, dst_subgroup, time_index, time_dimension_name, time)

def copy_time_sliced_nc_file(source_file:str, target_file,
                             time_index:np.ndarray,
                             time_varname:str,
                             time:datetime) -> None:
    # Open the source netCDF file in read mode
    with netCDF4.Dataset(source_file, 'r') as src:
        # Open the target netCDF file in write mode
        with netCDF4.Dataset(target_file, 'w') as dst:
            # Copy the root group (dimensions, variables, and groups)
            time_dimension_name = src[time_varname].dimensions[0]
            copy_group(src, dst, time_index, time_dimension_name, time)
            dst.sync()

if __name__ == '__main__':
    copy_time_sliced_nc_file('/bettik/yumengch-ext/DATA/25km_NH.TOPAZ4_2010-01-01_2011-01-01.nc',
                             'test.nc', np.arange(2), time_dimension_name='time', )