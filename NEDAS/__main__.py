from NEDAS.config import Config

def main() -> None:
    print("Running NEDAS")
    c = Config(parse_args=True)

    # prepare files
    # initial ensemble
    # truth

    # run analysis scheme
    c.scheme.run(c)

if __name__ == '__main__':
    main()
