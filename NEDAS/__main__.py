from NEDAS.config import Config
from NEDAS.schemes.get_analysis_scheme import get_analysis_scheme

def main() -> None:
    c = Config(parse_args=True)
    scheme = get_analysis_scheme(c)

    # prepare files
    # initial ensemble
    # truth

    # run analysis scheme
    print("Running NEDAS analysis scheme")
    scheme(c)

if __name__ == '__main__':
    main()
