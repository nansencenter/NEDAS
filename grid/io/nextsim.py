from pynextsim import NextsimBin

def open(filename):
    f = NextsimBin(filename)
    return f

def get_dat(f, varname):
    if varname in ('M_VT', 'M_UM', 'M_UT'):
        x = f.mesh_info.nodes_x
        y = f.mesh_info.nodes_y
        nnodes = f.mesh_info.num_nodes
        dat = f.get_var(varname)
        return x, y, dat[0:nnodes], dat[nnodes:2*nnodes]
    else:
        x, y = f.mesh_info.get_elements_xy()
        dat = f.get_var(varname)
        return x, y, dat

def get_gridded_dat(f, varname, x, y):
    dat = f.get_gridded_vars([varname], x, y)
    if varname in ('M_VT', 'M_UM', 'M_UT'):
        return dat[varname+'_1'], dat[varname+'_2']
    else:
        return dat[varname]

# def write(f, grid, varname, dat):

