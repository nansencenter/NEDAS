from NEDAS.config import Config
from NEDAS.schemes import get_scheme

def main() -> None:
    cf = Config(parse_args=True)
    scheme = get_scheme(cf)
    scheme()

if __name__ == '__main__':
    main()
