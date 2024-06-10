##from nextsim-tools/pynextsim

import numpy as np

##projection used in msh files
from pyproj import Proj
proj = Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)

class MeshPhysicalName:
    """
    Class used to distinguish coastal or open boundaries

    MeshPhysicalName(name,ident,topodim)

    Parameters:
    -----------
    name : string
        eg 'coast' or 'open'
    ident : int
        eg edge number
    topodim : int
        eg 2 (2D space) or 3 (3D space)
    """
    def __init__(self, name, ident, topodim):
        self.name = name
        self.ident = ident
        self.topodim = topodim

    def write(self, fid):
        '''
        add a physical name to the output file

        Parameters:
        -----------
        fid : _io.TextIOWrapper
        '''
        fid.write('%i %i "%s"\n' %(
            self.topodim, self.ident, self.name))

class MeshElement:
    """ Class used for both individual triangles and edges """

    def __init__(self, ident, eltype, tags, node_ids, node_indices):
        """
        Parameters:
        -----------
        ident : int
            element number
        eltype : int
            determines if triangle (2) or edge (1)
        vertices : list(int)
            list of mesh node ids
        tags : list
            tags from .msh files
            physical, elementary...
        node_ids : list(int)
            list of node ids
        node_indices : list(int)
            list of node indices
        """
        self.ident = ident
        self.eltype = eltype
        self.node_ids = node_ids
        self.node_indices = node_indices
        self.tags = tags
        self.physical = None
        if len(tags)>0:
            self.physical = tags[0] #to distinguish open/coast
        self.num_vertices = len(node_ids)

    def get_coords(self, xnod, ynod):
        """
        clist = self.get_coords(xnod,ynod)

        Parameters:
        -----------
        xnod : np.ndarray
            x coords of nodes
        ynod : np.ndarray
            y coords of nodes

        Returns:
        --------
        clist : list
            list of tuples with x,y coords of nodes for the element
        """
        return [(xnod[n], ynod[n]) for n in self.node_indices]

    def write(self, fid):
        '''
        add the element info to the output file

        Parameters:
        -----------
        fid : _io.TextIOWrapper
        '''
        lst = ['%i %i %i' %(
            self.ident, self.eltype, len(self.tags))]
        lst += ['%i' %t for t in self.tags]
        lst += ['%i' %i for i in self.node_ids]
        fid.write(" ".join(lst) + "\n")

class GmshBoundary:
    """
    Class for handling mesh boundaries
    """
    def __init__(self, exterior, islands=None,
            open_boundaries=None, coastal_boundaries=None):
        """
        Parameters:
        -----------
        exterior : shapely.geometry.Polygon
        islands : list(shapely.geometry.Polygon)
            - internal closed boundaries
        open_boundaries : list (shapely.geometry.LineString)
            - external open boundaries
        coastal_boundaries : list (shapely.geometry.LineString)
            - external closed boundaries
        """

        self.exterior_polygon = exterior
        self.island_polygons = islands
        self.open_boundaries = open_boundaries
        self.coastal_boundaries = coastal_boundaries

        xe, ye = exterior.exterior.xy
        self._set_xy_range(xe, ye)
        self._set_resolution(xe, ye)

    def _set_xy_range(self, xe, ye):
        """
        Set the x-y range

        Parameters:
        -----------
        xe : numpy.ndarray
            x coords of exterior polygon
        ye : numpy.ndarray
            y coords of exterior polygon

        Sets:
        -----
        self.xmin : float
        self.xmax : float
        self.ymin : float
        self.ymax : float
        """
        self.xmin = np.min(xe)
        self.xmax = np.max(xe)
        self.ymin = np.min(ye)
        self.ymax = np.max(ye)

    def _set_resolution(self, xe, ye):
        """
        Estimate the mesh resolution

        Parameters:
        -----------
        xe : numpy.ndarray
            x coords of exterior polygon
        ye : numpy.ndarray
            y coords of exterior polygon

        Sets:
        -----
        self.resolution : float
        """
        dx = np.diff(xe)
        dy = np.diff(ye)
        self.resolution = np.mean(np.hypot(dx, dy))

    def iswet(self, x, y):
        """
        use matplotlib.path to test if multiple points are contained
        inside the polygon self.exterior_polygon

        Parameters:
        -----------
        x: numpy.ndarray
            x coordinates to test
        y: numpy.ndarray
            y coordinates to test

        Returns:
        --------
        wet : numpy.ndarray(bool)
            mask of same shape as x and y
            - element is True/False if corresponding point is inside/outside the mesh
              (inside external poly but outside island ones)
        """
        coords = np.array([x.flatten(), y.flatten()]).T
        # test if inside external polygon
        mask = self.points_in_polygon(self.exterior_polygon, coords)
        # test not inside island polygons
        for p in self.island_polygons:
            mask *= ~self.points_in_polygon(p, coords)
        return mask.reshape(x.shape)

def xyz_to_lonlat(x, y, z):
    radius = np.array(np.sqrt(pow(x, 2.)+pow(y, 2.)+pow(z, 2.)), dtype=np.float64)
    r2d = 180./np.pi
    lat = r2d * np.arcsin(z / radius)
    lon = r2d * np.arctan2(y, x)
    lon -= 360. * np.floor(lon / 360.) # 0 <= lon < 360
    return lon, lat

def lonlat_to_xyz(lon, lat):
    d2r = np.pi / 180.
    rlat = d2r * lat
    rlon = d2r * lon
    z = np.sin(rlat)
    r = np.cos(rlat)
    x = r * np.cos(rlon)
    y = r * np.sin(rlon)
    return x, y, z

###functions to handle msh files
def read_mshfile(filename):
    info = {}
    with open(filename, 'r') as f:
        version, fmt, size = read_version_info(f)
        physical_names = read_physical_names(f)
        nodes_x, nodes_y, nodes_id, nodes_map = read_nodes(f)
        num_elements, edges, triangles = read_elements(f, nodes_map)
        info.update({'version':version, 'fmt':fmt, 'size':size,
                     'physical_names':physical_names,
                     'nodes_x':nodes_x, 'nodes_y':nodes_y,
                     'nodes_id':nodes_id, 'nodes_map':nodes_map,
                     'num_elements':num_elements,
                     'edges':edges, 'triangles':triangles,
                     })
        return info

def read_version_info(f):
    if "$MeshFormat" not in f.readline():
        raise ValueError("expecting $MeshFormat - not found")
    version, fmt, size = f.readline().split()
    if "$EndMeshFormat" not in f.readline():
        raise ValueError("expecting $EndMeshFormat - not found")
    return np.float32(version), int(fmt), int(size)

def read_physical_names(f):
    # GMSH meshes can have Physical Names defined at start
    physical_names = {}
    num_physical_names = 0
    if "$PhysicalNames" not in f.readline():
        raise ValueError("expecting $PhysicalNames -  not found")
    num_physical_names = int(f.readline())
    for _ in range(num_physical_names):
        topodim, ident, name = f.readline().split()
        if name[0]=='"':
            name  = name[1:-1]
        physical_names.update({int(ident):
                    MeshPhysicalName(name, int(ident), int(topodim))})
    if "$EndPhysicalNames" not in f.readline():
        raise ValueError("expecting $EndPhysicalNames - not found")
    return physical_names

def read_nodes(f):
    lin = f.readline()
    nodstr = lin.split()[0]
    if not ( ("$NOD" in lin)
            or ("$Nodes" in lin)
            or ("$ParametricNodes" in lin) ):
        msg = ("""Invalid nodes string '%s' in gmsh importer.
                    It should be either '$NOD','$Nodes'
                    or '$ParametricNodes'."""
                    %(nodstr))
        raise ValueError(msg)

    num_nodes = int(f.readline())
    lines = [f.readline().strip() for n in range(num_nodes)]
    iccc = np.array([[np.float64(v) for v in line.split()] for line in lines])
    lon, lat = xyz_to_lonlat(iccc[:, 1], iccc[:, 2], iccc[:, 3])
    nodes_x, nodes_y = proj(lon, lat)
    nodes_id = iccc[:,0].astype(int)
    nodes_map = {nodes_id[i]: i for i in range(nodes_id.size)}

    if ("$End"+nodstr[1:] not in f.readline()):
        raise ValueError("expecting $End"+nodstr[1:] +" - not found")
    return nodes_x, nodes_y, nodes_id, nodes_map

def parse_element_string(lin, nodes_map):
    slin = lin.split()
    elnumber, eltype, num_tags = [int(s) for s in slin[:3]]
    tags = [int(s) for s in slin[3:3+num_tags]]
    node_ids = [int(s) for s in slin[3+num_tags:]]
    node_inds = [nodes_map[nid] for nid in node_ids]
    el_obj = MeshElement(elnumber, eltype, tags, node_ids, node_inds)
    return el_obj

def read_elements(f, nodes_map):
    if '$Elements' not in f.readline():
        raise ValueError("invalid elements string in gmsh importer.")

    num_elements = int(f.readline())
    edges = []
    triangles = []
    for _ in range(num_elements):
        el_obj = parse_element_string(f.readline(), nodes_map)
        if el_obj.num_vertices==3:
            triangles.append(el_obj)
        else:
            edges.append(el_obj)
    num_edges     = len(edges)
    num_triangles = len(triangles)

    if ("$EndElements" not in f.readline()):
        raise ValueError("expecting $EndElements - not found")
    return num_elements, edges, triangles

def write_mshfile(filename, mesh_info):
    print('Saving %s' %filename)
    with open(filename, 'w') as f:
        write_version_info(f, mesh_info)
        write_physical_names(f, mesh_info)
        write_nodes(f, mesh_info)
        write_elements(f, mesh_info)

def write_version_info(f, mesh_info):
    version = mesh_info['version']
    fmt = mesh_info['fmt']
    size = mesh_info['size']
    f.write("$MeshFormat\n")
    f.write("%3.1f %i %i\n" %(version, fmt, size))
    f.write("$EndMeshFormat\n")

def write_physical_names(f, mesh_info):
    pnames = mesh_info['physical_names']
    f.write("$PhysicalNames\n")
    f.write("%i\n" %len(pnames))
    for pn in pnames.values():
        pn.write(f)
    f.write("$EndPhysicalNames\n")

def write_nodes(f, mesh_info):
    lon, lat = proj(mesh_info['nodes_x'], mesh_info['nodes_y'], inverse=True)
    x, y, z = lonlat_to_xyz(lon, lat)
    f.write("$Nodes\n")
    f.write("%i\n" %len(x))
    for i, xi, yi, zi in zip(mesh_info['nodes_id'], x, y, z):
        f.write("%i %18.16f %18.16f %18.16f\n" %(i, xi, yi, zi))
    f.write("$EndNodes\n")

def write_elements(f, mesh_info):
    f.write("$Elements\n")
    f.write("%i\n" %mesh_info['num_elements'])
    for e in mesh_info['edges']: e.write(f)
    for t in mesh_info['triangles']: t.write(f)
    f.write("$EndElements\n")

