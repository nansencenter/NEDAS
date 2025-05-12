import os
import subprocess
import numpy as np
from NEDAS.utils.conversion import t2s, dt1h
from ..util import read_data_bin, write_data_bin, grid2spec, spec2grid
from ..model import QGModel

class QGModelEmulator(QGModel):
    """
    Class for configuring and running the qg model emulator
    """
    def run_batch(self, nens=1, task_id=0, **kwargs):
        kwargs = super().super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        ##load model weights if not yet
        if self.unet_model is None:
            from .netutils import Att_Res_UNet
            self.unet_model = Att_Res_UNet(**self.model_params).make_unet_model()
            self.unet_model.load_weights(self.weightfile)

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

    def run(self, task_id=0, **kwargs):
        raise NotImplementedError("qg.emulator only runs in batch mode, call run_batch instead")
