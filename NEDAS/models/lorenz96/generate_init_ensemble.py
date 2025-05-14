import os
from NEDAS.config import Config

def main():
    c = Config(parse_args=True)

    model = c.models['lorenz96']

    ens_init_dir = model.ens_init_dir
    os.system("mkdir -p "+ens_init_dir)

    for m in range(c.nens):
        opts = { 'path': ens_init_dir, 'name': 'state', 'time': c.time, 'member': m, }
        print(f"generating initial condition for member {m+1}, output to {model.filename(**opts)}")

        state = model.generate_initial_condition()

        model.write_var(state, **opts)

    print("done.")

if __name__ == '__main__':
    main()
