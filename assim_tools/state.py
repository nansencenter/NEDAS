import numpy as np
import os
import struct
import importlib

from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t, dt1h
from utils.progress import print_with_cache, progress_bar
from utils.parallel import distribute_tasks, bcast_by_root, by_rank

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
    info = {'nx':c.nx, 'ny':c.ny, 'size':0, 'fields':{}, 'scalars':[]}
    rec_id = 0   ##record id for a 2D field
    pos = 0      ##seek position for rec
    variables = set()

    ##loop through variables in state_def
    for vrec in c.state_def:
        vname = vrec['name']
        variables.add(vname)

        if vrec['var_type'] == 'field':
            ##this is a state variable 'field' with dimensions t, z, y, x
            ##some properties of the variable is defined in its source module
            # src = importlib.import_module('models.'+vrec['model_src'])
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

        if vrec['var_type'] == 'scalar':
            ##this is a scalar (model parameter, etc.) to be updated
            ##since there is no difficulty storing the scalars on 1 proc
            ##we don't bother with parallelization (no rec_id needed)
            for time in c.time + np.array(c.state_time_steps)*dt1h:
                rec = {'name': vname,
                       'model_src': vrec['model_src'],
                       'err_type': vrec['err_type'],
                       'time': time,
                      }
                info['scalars'].append(rec)

    if c.debug:
        print(f"number of ensemble members, nens={c.nens}", flush=True)
        print(f"number of unique field records, nrec={len(info['fields'])}", flush=True)
        print(f"variables: {variables}", flush=True)

    info['size'] = pos ##size of a complete state (fields) for 1 memeber

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
        ##first line: some dimension sizes
        f.write('{} {} {}\n'.format(info['nx'], info['ny'], info['size']))

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

        ss = lines[0].split()
        info = {'nx':int(ss[0]), 'ny':int(ss[1]), 'size':int(ss[2]), 'fields':{}}

        ##records for uniq fields
        rec_id = 0
        for lin in lines[1:]:
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

    - mask: bool, np.array with shape (ny, nx)
      True if the grid point is masked (for example land grid point in ocean models).
      The masked points will not be stored in the binfile to reduce disk usage.

    - mem_id: int
      Member index, from 0 to nens-1

    - rec_id: int
      Field record index, info['fields'][rec_id] gives the record information

    - fld: float, np.array
      The field to be written to the file
    """
    ny = info['ny']
    nx = info['nx']
    rec = info['fields'][rec_id]

    fld_shape = (2, ny, nx) if rec['is_vector'] else (ny, nx)
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

    - mask: bool, np.array with shape (ny, nx)
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
    ny = info['ny']
    nx = info['nx']
    rec = info['fields'][rec_id]
    nv = 2 if rec['is_vector'] else 1

    fld_shape = (2, ny, nx) if rec['is_vector'] else (ny, nx)
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
    """
    Generate spatial partitioning of the domain
    partitions: dict[par_id, tuple(istart, iend, di, jstart, jend, dj)]
    for each partition indexed by par_id, the tuple contains indices for slicing the domain
    """
    if c.assim_mode == 'batch':
        ##divide into square tiles with nx_tile grid points in each direction
        ##the workload on each tile is uneven since there are masked points
        ##so we divide into 3*nproc tiles so that they can be distributed
        ##according to their load (number of unmasked points)
        nx_tile = np.maximum(int(np.round(np.sqrt(c.nx * c.ny / c.nproc_mem / 3))), 1)

        ##a list of (istart, iend, di, jstart, jend, dj) for tiles
        ##note: we have 3*nproc entries in the list
        partitions = [(i, np.minimum(i+nx_tile, c.nx), 1,   ##istart, iend, di
                       j, np.minimum(j+nx_tile, c.ny), 1)   ##jstart, jend, dj
                      for j in np.arange(0, c.ny, nx_tile)
                      for i in np.arange(0, c.nx, nx_tile) ]

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
        partitions = [(i, c.nx, nx_intv, j, c.ny, ny_intv)
                      for j in np.arange(ny_intv)
                      for i in np.arange(nx_intv) ]
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
    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('save state to '+state_file+'\n')

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
                print(progress_bar(m*nr+r, nm*nr))

            ##get the field record for output
            fld = fields[mem_id, rec_id]

            ##write the data to binary file
            write_field(state_file, c.state_info, c.mask, mem_id, rec_id, fld)

    if c.debug:
        print(' done.\n')


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

    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('compute ensemble mean, save to '+mean_file+'\n')
    if c.pid == 0:
        open(mean_file, 'wb')
        write_state_info(mean_file, c.state_info)
    c.comm.Barrier()

    for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
        if c.debug:
            print(progress_bar(r, len(c.rec_list[c.pid_rec])))

        ##initialize a zero field with right dimensions for rec_id
        if c.state_info['fields'][rec_id]['is_vector']:
            sum_fld_pid = np.zeros((2, c.ny, c.nx))
        else:
            sum_fld_pid = np.zeros((c.ny, c.nx))

        ##sum over all fields locally stored on pid
        for mem_id in c.mem_list[c.pid_mem]:
            sum_fld_pid += fields[mem_id, rec_id]

        ##sum over all field sums on different pids together to get the total sum
        ##TODO:reduce is expensive if only part of pid holds state in memory
        sum_fld = c.comm_mem.reduce(sum_fld_pid, root=0)

        if c.pid_mem == 0:
            mean_fld = sum_fld / c.nens
            write_field(mean_file, c.state_info, c.mask, 0, rec_id, mean_fld)

    if c.debug:
        print(' done.\n')

    ##clean up
    # del sum_fld_pid, sum_fld


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
    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('prepare state by reading fields from model restart\n')
    fields = {}
    z_coords = {}

    ##process the fields, each proc gets its own workload as a subset of
    ##mem_id,rec_id; all pid goes through their own task list simultaneously
    nm = len(c.mem_list[c.pid_mem])
    nr = len(c.rec_list[c.pid_rec])

    for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
        for r, rec_id in enumerate(c.rec_list[c.pid_rec]):
            if c.debug:
                print(progress_bar(m*nr+r, nm*nr))

            rec = c.state_info['fields'][rec_id]

            ##directory storing model output
            path = os.path.join(c.work_dir, 'cycle', t2s(rec['time']), rec['model_src'])

            ##the model object for handling this variable
            model = c.model_config[rec['model_src']]

            model.read_grid(path=path, member=mem_id, **rec)
            model.grid.set_destination_grid(c.grid)

            ##read field and save to dict
            var = model.read_var(path=path, member=mem_id, **rec)
            fld = model.grid.convert(var, is_vector=rec['is_vector'], method='linear', coarse_grain=True)
            fields[mem_id, rec_id] = fld

            ##misc. transform

            ##read z_coords for the field
            ##only need to generate the uniq z coords, store in bank
            zvar = model.z_coords(path=path, member=mem_id, **rec)
            z = model.grid.convert(zvar, is_vector=False, method='linear', coarse_grain=True)
            if rec['is_vector']:
                z_coords[mem_id, rec_id] = np.array([z, z])
            else:
                z_coords[mem_id, rec_id] = z
    if c.debug:
        print(' done.\n')
    c.comm.Barrier()

    return fields, z_coords


