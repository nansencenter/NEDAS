from NEDAS.config import Config
from NEDAS.schemes.get_analysis_scheme import get_analysis_scheme

def main():
    print("Running NEDAS")
    c = Config(parse_args=True)
    scheme = get_analysis_scheme(c)
    scheme(c)

if __name__ == '__main__':
    main()
