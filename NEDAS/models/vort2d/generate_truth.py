import os
from NEDAS.config import Config
from NEDAS.utils.conversion import dt1h

def main():
    c = Config(parse_args=True)

    model = c.models['vort2d']

    truth_dir = model.truth_dir
    os.system("mkdir -p "+truth_dir)

    t = c.time_start
    while t < c.time_end:
        opts = {
            'path': truth_dir,
            'name': 'velocity',
            'is_vector': True,
            'time': t,
            }

        if t == c.time_start:
            state = model.generate_initial_condition()
            print(f"generating initial condition {model.filename(**opts)}")
            model.write_var(state, **opts)

        next_t = t + c.cycle_period * dt1h
        print(f"running model, saving output {model.filename(**{**opts, 'time':next_t})}")
        model.run(path=truth_dir, time=t, forecast_period=c.cycle_period)

        t = next_t

    print("done.")

if __name__ == '__main__':
    main()
