import sys
from NEDAS import get_scheme

def main() -> None:
    try:
        scheme = get_scheme(parse_args=True)

        step = scheme.config.step
        if step:
            scheme.run_step(step)
            return

        scheme()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(1)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
