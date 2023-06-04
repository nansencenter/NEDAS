import numpy as np

type_convert = {'double':np.float64, 'float':np.float32, 'int':np.int32}
type_dic = {'double':'d', '8':'d', 'single':'f', 'float':'f', '4':'f', 'int':'i'}
type_size = {'double':8, 'float':4, 'int':4}

def variable_info(filename):
    v_info = {}
    with open(filename.replace('.bin','.dat'), 'r') as f:
        lines = f.readlines()
        pos = 4
        for recno, lin in enumerate(lines):
            ss = lin.split()
            if len(ss) < 2:
                continue
            ###get info about the variable record
            v_name = ss[0]
            v_type = ss[1]
            v_len = int(ss[2])
            typeconv = type_convert[v_type]
            v_min_val = typeconv(ss[3])
            v_max_val = typeconv(ss[4])
            v_rec = {}
            v_rec.update({'type':v_type})
            v_rec.update({'len':v_len})
            v_rec.update({'min_val':v_min_val})
            v_rec.update({'max_val':v_max_val})
            v_rec.update({'recno':recno})
            v_rec.update({'pos':pos})

            ##add variable record to the v_info dict
            v_info.update({v_name: v_rec})

            pos += (4 + v_len*type_size[v_type])
    return v_info

def read_var(filename, v_name):
    import struct
    v_info = variable_info(filename)

    if v_name not in v_info.keys():
        raise ValueError('variable %s not found in %s' %(v_name, filename))

    with open(filename, 'rb') as f:
        f.seek(v_info[v_name]['pos'])
        v_len = v_info[v_name]['len']
        v_type = v_info[v_name]['type']
        v_data = np.array(struct.unpack((v_len*type_dic[v_type]),
                                        f.read(v_len*type_size[v_type])))
    return v_data

def write_var(filename, v_name, v_rec, v_data):
    v_info = variable_info(filename)

    if v_name not in v_info.keys():
        raise ValueError('variable %s not in %s' %(v_name, filename))

    with open(filename.replace('.bin','.dat'), 'wt') as f:
        for v in v_info:
            if v_info[v]['recno'] == v_rec['recno']:
                if v_info[v]['type'] != v_rec['type'] or v_info[v]['len'] != v_rec['len']:
                    raise ValueError('attempt to overwrite file with mismatch data type/size')
                v_info[v] = v_rec ##new v_rec for v_data in dat_file
            ss = '%s %s %i' %(v, v_info[v]['type'], v_info[v]['len'])
            if v_info[v]['type'] == 'int':
                ss += ' %i %i' %(v_info[v]['min_val'], v_info[v]['max_val'])
            else:
                ss += ' %g %g' %(v_info[v]['min_val'], v_info[v]['max_val'])
            f.write(ss+'\n')

    with open(filename, 'r+b') as f:
        if len(v_data) != v_rec['len']:
            raise ValueError('v_data length does not match v_rec length')
        f.seek(v_info[v_name]['pos'])
        f.write(v_data.tobytes())

##nextsim map projection
from pyproj import Proj
proj = Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)

##nextsim mesh info
def indices(meshfile):
    elements = read_var(meshfile, 'Elements')
    ne = int(elements.size/3)
    ind = elements.reshape((ne, 3)) - 1
    return ind

def nodes_xy(meshfile):
    xn = read_var(meshfile, 'Nodes_x')
    yn = read_var(meshfile, 'Nodes_y')
    return xn, yn

def elements_xy(meshfile):
    xn, yn = nodes_xy(meshfile)
    ind = indices(meshfile)
    xe = np.mean(xn[ind], axis=1)
    ye = np.mean(yn[ind], axis=1)
    return xe, ye

def triangulation(meshfile):
    from matplotlib.tri import Triangulation
    xn, yn = nodes_xy(meshfile)
    ind = indices(meshfile)
    return Triangulation(xn, yn, ind)

def get_unstruct_grid_from_msh(msh_file):
    '''
    Get the unstructured grid from .msh files
    output: x[:], y[:], z[:]
    '''
    f = open(msh_file, 'r')
    if "$MeshFormat" not in f.readline():
        raise ValueError("expecting $MeshFormat -  not found")
    version, fmt, size = f.readline().split()
    if "$EndMeshFormat" not in f.readline():
        raise ValueError("expecting $EndMeshFormat -  not found")
    if "$PhysicalNames" not in f.readline():
        raise ValueError("expecting $PhysicalNames -  not found")
    num_physical_names = int(f.readline())
    for _ in range(num_physical_names):
        topodim, ident, name = f.readline().split()
    if "$EndPhysicalNames" not in f.readline():
        raise ValueError("expecting $EndPhysicalNames -  not found")
    if "$Nodes" not in f.readline():
        raise ValueError("expecting $Nodes -  not found")
    num_nodes = int(f.readline())
    lines = [f.readline().strip() for n in range(num_nodes)]
    iccc =np.array([[float(v) for v in line.split()] for line in lines])
    x, y, z = (iccc[:, 1], iccc[:, 2], iccc[:, 3])
    if "$EndNodes" not in f.readline():
        raise ValueError("expecting $EndNodes -  not found")
    return x, y, z

##nextsim variable names
variable_dic = {}

def get_var(filename, v_name):
    if v_name == "M_VT":
        return read_var(filename, v_name).reshape((2, -1))
    else:
        return read_var(filename, v_name)
