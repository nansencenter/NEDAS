import os
from NEDAS.config import Config

def generate_init_ensemble(c: Config) -> None:
    model = c.models['vort2d']

    ens_init_dir = model.ens_init_dir
    os.system("mkdir -p "+ens_init_dir)

    for m in range(c.nens):
        opts = {
            'path': ens_init_dir,
            'name': 'velocity',
            'is_vector': True,
            'time': c.time,
            'member': m,
            }
        print(f"generating initial condition for member {m+1}, output to {model.filename(**opts)}")

        state = model.generate_initial_condition()

        model.write_var(state, **opts)

    print("done.")

if __name__ == '__main__':
    c = Config(parse_args=True)
    generate_init_ensemble()
