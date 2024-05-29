from datetime import datetime as dt

from models.qg import Model

from config import Config

c = Config()
m = Model()
m.run(1, c, '/cluster/work/users/yingyue/qg/run', time=dt(2001,1,1), member=0)

