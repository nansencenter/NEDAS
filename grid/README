Ying 2022

Generating a reference grid for DA state vector

polar region uses stereographic projection: lat/lon -> x, y, z

generate uniform grid x,y, centered on North Pole, in meter units

gen_reference_grid.py makes proper reference_grid.nc file, containing
    plat(x, y)          #latitude
    plon(x, y)          #longitude
    ptheta(x, y)        #rotation angle for vectors
    x_corners(x, y, 4)  #4 corner coordinate values for mesh nodes (for conservative mapping in interpolation)
    y_corners(x, y, 4)
    z_corners(x, y, 4)

these variables are used in [reference_grid.nc] for nextsim to output nc format [statevector]

