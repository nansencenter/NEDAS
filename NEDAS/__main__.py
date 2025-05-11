from NEDAS.config import Config

def main() -> None:
    print("Running NEDAS")
    c = Config(parse_args=True)
    c.scheme.run()

if __name__ == '__main__':
    main()
