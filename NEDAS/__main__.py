from NEDAS.config import Config
from NEDAS.schemes import get_scheme

def main() -> None:
    config = Config(parse_args=True)
    scheme = get_scheme(config)
    scheme()

if __name__ == '__main__':
    main()
