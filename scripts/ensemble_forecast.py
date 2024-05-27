import importlib
import time

from utils.parallel import run_by_root
from utils.log import message, timer


def ensemble_forecast(c):
    """
    This function runs ensemble forecasts to advance to the next cycle
    """

    message(c.comm, f"ensemble forecast by {c.pid}")
    for mem_id in range(c.nens):
        message(c.comm, f"{mem_id} ", c.pid_show)
        time.sleep(1)
        # for model_config in c.model_def:

            ##load the model module and call run_model
            # model_src = importlib.import_module('models.'+model_config['name'])

            # model_src.run_model(member=mem_id, **model_config)


    message(c.comm, ' ensemble forecast complete\n\n', c.pid_show)


if __name__ == "__main__":

    from config import Config
    c = Config()

    timer(c.comm, c.pid_show)(run_by_root(c.comm)(ensemble_forecast))(c)

