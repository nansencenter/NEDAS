import config as c
import assim_tools

comm = assim_tools.parallel_start()

for s in range(c.nscale):  ##scale loop

    ##scale component stored in s_dir if multiscale
    if c.nscale > 1:
        s_dir = f'/s{c.scale+1}'
    else:
        s_dir = ''

    prior_state_file = c.work_dir+'/analysis/'+c.time+s_dir+'/state.bin'
    state = assim_tools.process_state(c, comm, prior_state_file)

    ##get a list of horizontal location ind from analysis grid
    ##the loc_list is to be distributed to processors based on their workload
    # ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))



    ##process observations
    #obs_seq_file = c.work_dir+'/analysis/'+c.time+s_dir+'/obs_seq.bin'

    ##analysis grid
    # grid = c.grid
    # nx = grid.nx
    # ny = grid.ny

    ##divide domain into 3*nproc square tiles
    ##number of grid points in one direction for each tile
    # nx_tile = int(np.round(np.sqrt(ny * nx / nproc / 3)))

    ##corners = (istart, iend, jstart, jend) for the tiles
    # corners = [(i, np.minimum(i+nx_tile, nx), j, np.minimum(j+nx_tile, ny))
                # for i in np.arange(0, nx, nx_tile) for j in np.arange(0, ny, nx_tile)]


    #state_full = 

    ##count number of state and obs in each chunk
    #state_count_tile = np.sum((~c.mask[:]).astype(int))

    #obs_count_tile = 1 + obs

    ##distribute chk to proc based on their load = state_count*obs_count
    #chk_list_proc = assim_tools.distribute_tasks(comm, chk_list, state_count_chk*obs_count_chk)

    #state = assim_tools.process_state(c, comm)


    ##save a copy of prior states in binary file
    #assim_tools.output_state(comm, state, state_file)


