##model parameters defined here
##if not defined in config, use default value

##grid dimensions
nx = 128
ny = 128

##grid spacing, in meters
dx = 9000

##time step in seconds
dt = 60

##model restart output interval in hours
restart_dt = 1

##model phys param
gen = 2e-5    ##vorticity generation rate
diss = 3e3    ##dissipation rate

### vortex parameters
Vmax = 35    ## maximum wind speed (vortex intensity), m/s
Rmw = 45000  ## radius of maximum wind (vortex size), m

##background flow
Vbg = 5      ##background flow wind speed (std), m/s
Vslope = -3  ##background flow kinetic energy spectrum power law

##some parameters for experiment setup
loc_sprd = 30000  ##initial spread in vortex position, m

