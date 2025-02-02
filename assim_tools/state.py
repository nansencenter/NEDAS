import numpy as np
import os
import struct
from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t, dt1h, ensure_list
from utils.progress import print_with_cache, progress_bar
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.multiscale import get_scale_component
from utils.dir_def import forecast_dir

"""
Note: The analysis is performed on a regular grid.
The entire state has dimensions: member, variable, time,  z,  y,  x
                     indexed by: mem_id,        v,    t,  k,  j,  i
                      with size:   nens,       nv,   nt, nz, ny, nx

To parallelize workload, we group the dimensions into 3 indices:
mem_id indexes the ensemble members
rec_id indexes the uniq 2D fields with (v, t, k), since nz and nt may vary
         for different variables, we stack these dimensions in the 'record'
         dimension with size nrec
par_id indexes the spatial partitions, which are subset of the 2D grid
         given by (ist, ied, di, jst, jed, dj), for a complete field fld[j,i]
         the processor with par_id stores fld[ist:ied:di, jst:jed:dj] locally.
"""
def parse_state_info(c):
    """
    Parses info for the nrec fields in the state.

    Input:
    - c: config obj with the environment variables

    Returns:
    - info: dict
      A dictionary with some dimensions and list of unique field records
    """
    info = {}
    info['size'] = 0
    info['shape'] = c.grid.x.shape
    info['fields'] = {}
    info['scalars'] = {}
    rec_id = 0   ##record id for a 2D field
    pos = 0      ##seek position for rec
    variables = set()
    err_types = set()

    ##loop through variables in state_def
    for vrec in ensure_list(c.state_def):
        vname = vrec['name']
        variables.add(vname)
        err_types.add(vrec['err_type'])

        if vrec['var_type'] == 'field':
            ##this is a state variable 'field' with dimensions t, z, y, x
            ##some properties of the variable is defined in its source module
            src = c.model_config[vrec['model_src']]
            assert vname in src.variables, 'variable '+vname+' not defined in '+vrec['model_src']+' Model.variables'

            #now go through time and zlevels to form a uniq field record
            for time in c.time + np.array(c.state_time_steps)*dt1h:
                for k in src.variables[vname]['levels']:
                    rec = { 'name': vname,
                            'model_src': vrec['model_src'],
                            'dtype': src.variables[vname]['dtype'],
                            'is_vector': src.variables[vname]['is_vector'],
                            'units': src.variables[vname]['units'],
                            'err_type': vrec['err_type'],
                            'time': time,
                            'dt': c.state_time_scale,
                            'k': k,
                            'pos': pos, }
                    info['fields'][rec_id] = rec

                    ##update seek position
                    nv = 2 if rec['is_vector'] else 1
                    fld_size = np.sum((~c.mask).astype(int))
                    pos += nv * fld_size * type_size[rec['dtype']]
                    rec_id += 1

        elif vrec['var_type'] == 'scalar':
            pass

        else:
            raise NotImplementedError(f"{vrec['var_type']} is not supported in the state vector.")

    if c.debug:
        print(f"number of ensemble members, nens={c.nens}", flush=True)
        print(f"number of unique field records, nrec={len(info['fields'])}", flush=True)
        print(f"variables: {variables}", flush=True)

    info['size'] = pos ##size of a complete state (fields) for 1 memeber
    info['variables'] = list(variables)
    info['err_types'] = list(err_types)
    return info

def write_state_info(binfile, info):
    """
    Write state_info to a .dat file accompanying the .bin file

    Inputs:
    - binfile: str
      File path for the .bin file

    - info: state_info
    """
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        ##first line: grid dimension
        if len(info['shape']) == 1:
            f.write('{}\n'.format(info['shape'][0]))
        else:
            f.write('{} {}\n'.format(info['shape'][0], info['shape'][1]))

        ##second line: total size of the state
        f.write('{}\n'.format(info['size']))

        ##followed by nfield lines: each for a field record
        for i, rec in info['fields'].items():
            name = rec['name']
            model_src = rec['model_src']
            dtype = rec['dtype']
            is_vector = int(rec['is_vector'])
            units = rec['units']
            err_type = rec['err_type']
            time = t2h(rec['time'])
            dt = rec['dt']
            k = rec['k']
            pos = rec['pos']
            f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(name, model_src, dtype, is_vector, units, err_type, time, dt, k, pos))

def read_state_info(binfile):
    """
    Read .dat file accompanying the .bin file and obtain state_info

    Input:
    - binfile: str
      File path for the .bin file

    Returns:
    - info: state_info dict
    """
    with open(binfile.replace('.bin','.dat'), 'r') as f:
        lines = f.readlines()
        info = {}

        ss = lines[0].split()
        if len(ss)==1:
            info['shape'] = (int(ss),)
        else:
            info['shape'] = (int(ss[0]), int(ss[1]))

        info['size'] = int(lines[1])

        ##records for uniq fields
        info['fields'] = {}
        rec_id = 0
        for lin in lines[2:]:
            ss = lin.split()
            rec = {'name': ss[0],
                   'model_src': ss[1],
                   'dtype': ss[2],
                   'is_vector': bool(int(ss[3])),
                   'units': ss[4],
                   'err_type': ss[5],
                   'time': h2t(np.float32(ss[6])),
                   'dt': np.float32(ss[7]),
                   'k': np.float32(ss[8]),
                   'pos': int(ss[9]), }
            info['fields'][rec_id] = rec
            rec_id += 1

    return info

def write_field(binfile, info, mask, mem_id, rec_id, fld):
    """
    Write a field to a binary file

    Inputs:
    - binfile: str
      File path for the .bin file

    - info: state_info dict

    - mask: bool, np.array with grid.x.shape
      True if the grid point is masked (for example land grid point in ocean models).
      The masked points will not be stored in the binfile to reduce disk usage.

    - mem_id: int
      Member index, from 0 to nens-1

    - rec_id: int
      Field record index, info['fields'][rec_id] gives the record information

    - fld: float, np.array
      The field to be written to the file
    """
    rec = info['fields'][rec_id]

    fld_shape = (2,)+info['shape'] if rec['is_vector'] else info['shape']
    assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

    if rec['is_vector']:
        fld_ = fld[:, ~mask].flatten()
    else:
        fld_ = fld[~mask]

    with open(binfile, 'r+b') as f:
        f.seek(mem_id*info['size'] + rec['pos'])
        f.write(struct.pack(fld_.size*type_dic[rec['dtype']], *fld_))

def read_field(binfile, info, mask, mem_id, rec_id):
    """
    Read a field from a binary file

    Inputs:
    - binfile: str
      File path for the .bin file

    - info: state_info dict

    - mask: bool, np.array with grid.x.shape
      True if the grid point is masked (for example land grid point in ocean models).
      The masked points will not be stored in the binfile to reduce disk usage.

    - mem_id: int
      Member index from 0 to nens-1

    - rec_id: int
      Field record index, info['fields'][rec_id] gives the record information

    Returns:
    - fld: float, np.array
      The field read from the file
    """
    rec = info['fields'][rec_id]

    nv = 2 if rec['is_vector'] else 1
    fld_shape = (2,)+info['shape'] if rec['is_vector'] else info['shape']
    fld_size = np.sum((~mask).astype(int))

    with open(binfile, 'rb') as f:
        f.seek(mem_id*info['size'] + rec['pos'])
        fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec['dtype']]),
                        f.read(nv*fld_size*type_size[rec['dtype']])))
        fld = np.full(fld_shape, np.nan)
        if rec['is_vector']:
            fld[:, ~mask] = fld_.reshape((2, -1))
        else:
            fld[~mask] = fld_
        return fld

def distribute_state_tasks(c):
    """
    Distribute mem_id and rec_id across processors

    Inputs:
    - c: config module

    Returns:
    - mem_list: dict[pid_mem, list[mem_id]]
    - rec_list: dict[pid_rec, list[rec_id]]
    """
    ##list of mem_id as tasks
    mem_list = distribute_tasks(c.comm_mem, [m for m in range(c.nens)])

    ##list rec_id as tasks
    rec_list_full = [i for i in c.state_info['fields'].keys()]
    rec_size = np.array([2 if r['is_vector'] else 1 for i,r in c.state_info['fields'].items()])
    rec_list = distribute_tasks(c.comm_rec, rec_list_full, rec_size)

    return mem_list, rec_list

def partition_grid(c):
    if len(c.grid.x.shape)==2:
        return partition_regular_slicing(c)
    else:
        return partition_grid_point_list(c)

def partition_regular_slicing(c):
    """
    Generate spatial partitioning of the domain
    partitions: dict[par_id, tuple(istart, iend, di, jstart, jend, dj)]
    for each partition indexed by par_id, the tuple contains indices for slicing the domain
    Using regular slicing is more efficient than fancy indexing (used in irregular grid)
    """
    ny, nx = c.grid.x.shape

    if c.assim_mode == 'batch':
        ##divide into square tiles with nx_tile grid points in each direction
        ##the workload on each tile is uneven since there are masked points
        ##so we divide into 3*nproc tiles so that they can be distributed
        ##according to their load (number of unmasked points)
        ntile = c.nproc_mem * 3
        nx_tile = np.maximum(int(np.round(np.sqrt(nx * ny / ntile))), 1)

        ##a list of (istart, iend, di, jstart, jend, dj) for tiles
        ##note: we have 3*nproc entries in the list
        partitions = [(i, np.minimum(i+nx_tile, nx), 1,   ##istart, iend, di
                       j, np.minimum(j+nx_tile, ny), 1)   ##jstart, jend, dj
                      for j in np.arange(0, ny, nx_tile)
                      for i in np.arange(0, nx, nx_tile) ]

    elif c.assim_mode == 'serial':
        ##the domain is divided into tiles, each is formed by nproc_mem elements
        ##each element is stored on a different pid_mem
        ##for each pid, its loc points cover the entire domain with some spacing

        ##list of possible factoring of nproc_mem = nx_intv * ny_intv
        ##pick the last factoring that is most 'square', so that the interval
        ##is relatively even in both directions for each pid
        nx_intv, ny_intv = [(i, int(c.nproc_mem / i))
                            for i in range(1, int(np.ceil(np.sqrt(c.nproc_mem))) + 1)
                            if c.nproc_mem % i == 0][-1]

        ##a list of (ist, ied, di, jst, jed, dj) for slicing
        ##note: we have nproc_mem entries in the list
        partitions = [(i, nx, nx_intv, j, ny, ny_intv)
                      for j in range(ny_intv) for i in range(nx_intv) ]
    return partitions

def partition_grid_point_list(c):
    """
    Generate spatial partitioning of the domain
    partitions: dict[par_id, tuple(start, end, interval)]
    """
    npoints = c.grid.x.size
    if c.assim_mode == 'batch':
        ##divide the domain into sqaure tiles, similar to regular_grid case, but collect
        ##the grid points inside each tile and return the indices
        ntile = c.nproc_mem * 3

        if c.grid.Ly==0:
            ##for 1D grid, just divide into equal sections, no y dimension
            Dx = c.grid.Lx / ntile
            partitions = [np.where(np.logical_and(c.grid.x>=x, c.grid.x<x+Dx))[0]
                          for x in np.arange(c.grid.xmin, c.grid.xmax, Dx)]

        else:
            ##for 2D grid, find number of tiles in each direction according to aspect ratio
            ntile_y = max(int(np.sqrt(ntile * c.grid.Ly / c.grid.Lx)), 1)
            ntile_x = max(ntile // ntile_y, 1)
            Dx = c.grid.Lx / ntile_x
            Dy = c.grid.Ly / ntile_y
            partitions = [np.where(np.logical_and(np.logical_and(c.grid.x>=x, c.grid.x<x+Dx),
                                                  np.logical_and(c.grid.y>=y, c.grid.y<y+Dy)))
                          for y in np.arange(c.grid.ymin, c.grid.ymax, Dy)
                          for x in np.arange(c.grid.xmin, c.grid.xmax, Dx)]

    elif c.assim_mode == 'serial':
        ##just divide the list of points into nproc_mem parts, each part spanning the entire domain
        nparts = c.nproc_mem
        partitions = [np.arange(i, npoints, nparts) for i in np.arange(nparts)]

    return partitions

def output_state(c, fields, state_file):
    """
    Parallel output the fields to the binary state_file

    Inputs:
    - c: config module
    - fields: dict[(mem_id, rec_id), fld]
      the locally stored field-complete fields for output
    - state_file: str
      path to the output binary file
    """
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    print_1p('>>> save state to '+state_file+'\n')

    if c.pid == 0:
        ##if file doesn't exist, create the file
        open(state_file, 'wb')
        ##write state_info to the accompanying .dat file
        write_state_info(state_file, c.state_info)
    c.comm.Barrier()

    nm = len(c.mem_list[c.pid_mem])
    nr = len(c.rec_list[c.pid_rec])
    for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
        for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
            if c.debug:
                rec = c.state_info['fields'][rec_id]
                print(f"PID {c.pid:4}: saving field: mem{mem_id+1:03} '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
            else:
                print_1p(progress_bar(m*nr+r, nm*nr))

            ##get the field record for output
            fld = fields[mem_id, rec_id]

            ##write the data to binary file
            write_field(state_file, c.state_info, c.mask, mem_id, rec_id, fld)
    c.comm.Barrier()
    print_1p(' done.\n')

def output_ens_mean(c, fields, mean_file):
    """
    Compute ensemble mean of a field stored distributively on all pid_mem
    collect means on pid_mem=0, and output to mean_file

    Inputs:
    - c: config module
    - fields, dict[(mem_id, rec_id), fld]
      the locally stored field-complete fields for output
    - mean_file: str
      path to the output binary file for the ensemble mean
    """
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    print_1p('>>> compute ensemble mean, save to '+mean_file+'\n')

    if c.pid == 0:
        ##if file doesn't exist, create the file, write state_info
        open(mean_file, 'wb')
        write_state_info(mean_file, c.state_info)
    c.comm.Barrier()

    for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
        rec = c.state_info['fields'][rec_id]
        if c.debug:
            print(f"PID {c.pid:4}: saving mean field '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
        else:
            print_1p(progress_bar(r, len(c.rec_list[c.pid_rec])))

        ##initialize a zero field with right dimensions for rec_id
        fld_shape = (2,)+c.state_info['shape'] if rec['is_vector'] else c.state_info['shape']
        sum_fld_pid = np.zeros(fld_shape)

        ##sum over all fields locally stored on pid
        for mem_id in c.mem_list[c.pid_mem]:
            sum_fld_pid += fields[mem_id, rec_id]

        ##sum over all field sums on different pids together to get the total sum
        ##TODO:reduce is expensive if only part of pid holds state in memory
        sum_fld = c.comm_mem.reduce(sum_fld_pid, root=0)

        if c.pid_mem == 0:
            mean_fld = sum_fld / c.nens
            write_field(mean_file, c.state_info, c.mask, 0, rec_id, mean_fld)
    c.comm.Barrier()
    print_1p(' done.\n')

def prepare_state(c):
    """
    Collects fields from model restart files, convert them to the analysis grid,
    preprocess (coarse-graining etc), save to fields[mem_id, rec_id] pointing to the uniq fields

    Inputs:
    - c: config object

    Returns:
    - fields: dict[(mem_id, rec_id), fld]
      where fld is np.array defined on c.grid, it's one of the state variable field
    - z_coords: dict[(mem_id, rec_id), zfld]
      where zfld is same shape as fld, it's he z coordinates corresponding to each field
    """

    pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in c.rec_list.items() if len(lst)>0][0]
    c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show

    ##pid_show has some workload, it will print progress message
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    print_1p('>>> prepare state by reading fields from model restart\n')
    fields = {}
    z_coords = {}

    ##process the fields, each proc gets its own workload as a subset of
    ##mem_id,rec_id; all pid goes through their own task list simultaneously
    nm = len(c.mem_list[c.pid_mem])
    nr = len(c.rec_list[c.pid_rec])

    for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
        for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
            rec = c.state_info['fields'][rec_id]

            if c.debug:
                print(f"PID {c.pid:4}: prepare_state mem{mem_id+1:03} '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
            else:
                print_1p(progress_bar(m*nr+r, nm*nr))

            ##directory storing model output
            path = forecast_dir(c, rec['time'], rec['model_src'])

            ##the model object for handling this variable
            model = c.model_config[rec['model_src']]

            model.read_grid(path=path, member=mem_id, **rec)
            model.grid.set_destination_grid(c.grid)

            ##read field from restart file
            var = model.read_var(path=path, member=mem_id, **rec)
            fld = model.grid.convert(var, is_vector=rec['is_vector'], method='linear', coarse_grain=True)

            ##misc. transform can be added here
            ##e.g., multiscale approach
            if c.nscale > 1:
                ##get scale component for multiscale approach
                fld = get_scale_component(c.grid, fld, c.character_length, c.scale_id)

            ##save field to dict
            fields[mem_id, rec_id] = fld

            ##read z_coords for the field
            ##only need to generate the uniq z coords, store in bank
            zvar = model.z_coords(path=path, member=mem_id, **rec)
            z = model.grid.convert(zvar, is_vector=False, method='linear', coarse_grain=True)
            if rec['is_vector']:
                z_coords[mem_id, rec_id] = np.array([z, z])
            else:
                z_coords[mem_id, rec_id] = z
    c.comm.Barrier()
    print_1p(' done.\n')

    ##additonal output of debugging
    if c.debug:
        np.save(os.path.join(c.analysis_dir, f'fields_prior.{c.pid_mem}.{c.pid_rec}.npy'), fields)

    return fields, z_coords

