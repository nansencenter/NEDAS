from NEDAS.schemes import get_scheme

def main() -> None:
    scheme = get_scheme(parse_args=True)

    step = scheme.config.step
    if step:
        scheme.run_step(step)
        return

    scheme()

if __name__ == '__main__':
    main()
