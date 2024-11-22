##diagnostic variables getters
def get_seaice_drift(path, grid, **kwargs):
    return var

def get_seaice_deform_shear(path, grid, **kwargs):
    return var

variables = {
    'seaice_drift': {'getter':get_seaice_drift, 'dtype':'float', 'is_vector':True, 'levels':[0], 'units':'km/day' },
    'seaice_deform_shear': {'getter':get_seaice_drift, 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'1/day' },
    }
