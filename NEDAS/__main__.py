from NEDAS.config import Config
from NEDAS.schemes.get_analysis_scheme import get_analysis_scheme

def main():
    print("Running NEDAS")
    c = Config(parse_args=True)
    scheme = get_analysis_scheme(c)
    if c.step:
        if c.step in ['perturb', 'filter', 'diagnose']:
            mpi = True
        else:
            mpi = False
        scheme.run_step(c, c.step, mpi=mpi)
    else:
        scheme(c)

if __name__ == '__main__':
    main()
