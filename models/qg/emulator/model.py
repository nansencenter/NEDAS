import os
import subprocess
import numpy as np
from config import parse_config
from grid import Grid
from utils.conversion import t2s, dt1h
from ..util import read_data_bin, write_data_bin, grid2spec, spec2grid
from ..util import psi2zeta, psi2u, psi2v, psi2temp, uv2zeta, zeta2psi, temp2psi
from ..model import QGModel
from .netutils import Att_Res_UNet

class QGModelEmulator(QGModel):
    """
    Class for configuring and running the qg model emulator
    """
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.restart_dt = 12  ##the model is trained with this output interval
        self.unet_model = Att_Res_UNet(**self.model_params).make_unet_model()
        self.unet_model.load_weights(self.weightfile)

    def run_batch(self, nens=1, task_id=0, **kwargs):
        kwargs = super().super().parse_kwargs(**kwargs)
        self.run_status = 'running'
        if nens>1:
            ##running ensmeble together, ignore kwargs['member']
            members = list(range(nens))
        else:
            members = [kwargs['member']]

        path = kwargs['path']
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        nsteps = int(forecast_period / self.restart_dt)
        next_time = time + forecast_period * dt1h

        ##input ensemble state
        state = np.zeros((nens,self.ny,self.nx,self.nz))
        for m in range(nens):
            kwargs_in = {**kwargs, 'member':members[m], 'time':time}
            input_file = self.filename(**kwargs_in)
            run_dir = os.path.dirname(input_file)
            subprocess.run("mkdir -p "+run_dir, shell=True)

            input_file = self.filename(**kwargs_in)
            for k in range(self.nz):
                psik = read_data_bin(input_file, self.kmax, self.nz, k)
                state[m,...,k] = spec2grid(psik).T

        ##run prediction model
        for i in range(nsteps):
            state = self.unet_model.predict(state, verbose=1)
        state_out = state

        ##output to restart file
        for m in range(nens):
            kwargs_out = {**kwargs, 'member':members[m], 'time':next_time}
            output_file = self.filename(**kwargs_out)
            if not os.path.exists(output_file):
                with open(output_file, 'wb'):
                    pass
            for k in range(self.nz):
                fld = state_out[m,...,k]
                self.write_var(fld, name='streamfunc', k=k, **kwargs_out)
