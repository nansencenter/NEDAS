import sys
import traceback
from NEDAS import get_scheme
from NEDAS.utils.parallel import abort_all_ranks

def main() -> None:
    scheme = None
    try:
        scheme = get_scheme(parse_args=True)

        step = scheme.config.step
        if step:
            scheme.run_step(step)
            return

        scheme()

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
        abort_all_ranks(scheme.c.comm if scheme is not None else None, 1)

    except Exception:
        traceback.print_exc()
        abort_all_ranks(scheme.c.comm if scheme is not None else None, 1)

if __name__ == '__main__':
    main()
