##the map projection used in nextsim
import pyproj
proj = pyproj.Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)

from .basic_io import get_info, read_data, write_data
from matplotlib.tri import Triangulation
from grid import Grid

def indices(filename):
    elements = read_data(filename.replace('field','mesh'), 'Elements')
    ne = int(elements.size/3)
    ind = elements.reshape((ne, 3)) - 1
    return ind

def nodes_xy(filename):
    xn = read_data(filename.replace('field','mesh'), 'Nodes_x')
    yn = read_data(filename.replace('field','mesh'), 'Nodes_y')
    return xn, yn

def elements_xy(filename):
    xn, yn = nodes_xy(filename)
    ind = indices(filename)
    xe = np.mean(xn[ind], axis=1)
    ye = np.mean(yn[ind], axis=1)
    return xe, ye

def triangulation(filename):
    xn, yn = nodes_xy(filename)
    ind = indices(filename)
    return Triangulation(xn, yn, ind)

def get_grid(filename):
    x, y = nodes_xy(filename)
    tri = triangulation(filename)
    return Grid(proj, x, y, regular=False, triangles=tri.triangles)

##other funcs
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
