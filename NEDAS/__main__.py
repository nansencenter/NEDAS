from NEDAS.schemes import get_scheme

def main() -> None:
    scheme = get_scheme(parse_args=True)
    scheme()

if __name__ == '__main__':
    main()
