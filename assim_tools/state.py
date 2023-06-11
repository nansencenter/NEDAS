def prepare_state(c, time):
    ##input: c is the config module
    ##state_config, prepared by config.state_def func
    ##ref_grid, reference grid, prepared by config.ref_grid
    ##  nens members in the ensemble, each member has a separate state (output bin file)
    ##  state[nens, nfield, ny, nx], nfield dimension contains nv,nt,nz flattened
    ##  nv is number of variables, nt is time slices, nz is vertical layers,
    ##  of course nt,nz vary for each variables, so we stack them in nfield dimension

    import os
    import numpy as np
    import importlib
    from mpi4py import MPI
    from grid import Converter

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    proc_id = comm.Get_rank()
    ny, nx = c.ref_grid.x.shape

    if proc_id == 0:
        field_info = []
        fld_size = nx * ny * np.dtype(np.float64).itemsize
        field_id = 0
        pos = 0
        ##loop over variables in state_def to log field_info
        for varname in c.state_def:
            source = c.state_def[varname]['source']
            nz = c.state_def[varname]['nz']
            nt = c.state_def[varname]['nt']
            src = importlib.import_module(source)
            assert varname in src.variables, "variable "+varname+" not listed in "+source+".variables"
            for t in range(nt):
                for z in range(nz):
                    field_rec = {'id':field_id,
                                 'varname':varname,
                                 'source':source,
                                 'time':time,
                                 'level':z,
                                 'pos':pos }
                    field_info.append(field_rec)
                    field_id += 1
                    if src.variables[varname]['is_vector']:
                        pos += 2*fld_size
                    else:
                        pos += fld_size
    else:
        field_info = None
    field_info = comm.bcast(field_info, root=0)

    ##now actually get the fields into state vector
    # ##parallel work by all processors
    # proc_ens_id = ##proc id in nens dimension
    # proc_field_id = ##proc id in nfield dimension
    # ens_comm = comm.Split(proc_ens_id, proc_field_id)
    # nens_part = 
    # nfield_part = nfield

    # for field in field_info:
    field = field_info[proc_id]
    print(proc_id, field)
    src = importlib.import_module(field['source'])
    t = field['time']
    z = field['level']
    vname = field['varname']
    mem = 0
    fname = src.filename(c.DATA_DIR, mem, t)
    cnv = Converter(src.get_grid(fname), c.ref_grid)
    data_ = cnv.convert(src.get_var(fname, vname),
                        is_vector=src.variables[vname]['is_vector'])
    with open('1.bin', 'r+b') as f:
        f.seek(field['pos'])
        f.write(data_.tobytes())


if __name__ == '__main__':
    import config
    from datetime import datetime
    current_time = datetime(2007, 1, 20)
    prepare_state(config, current_time)
