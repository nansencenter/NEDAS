"""
Cost of crossing the Python/compiled boundary, for the assimilators that call compiled
kernels: PDAF through pyPDAF.

NEDAS's native assimilators and the compiled ones are run against each other to see what an
implementation's own choices cost. That comparison is only readable if the price of the
binding is known rather than guessed, and it is not a thin wrapper: PDAF calls *back* into
Python once per local analysis domain, so the crossing count grows with the problem.

**Off unless asked for.** The regions measured here wrap a few hundred nanoseconds of work,
so an instrument costing the same would both slow the analysis and inflate the number it
reports. Disabled -- the default -- ``wrap`` hands the function straight back and ``measure``
returns a shared do-nothing region, which is why ``Context.timer`` is not used for this: it
barriers on every call (a collective op in a per-domain callback deadlocks when ranks hold
different numbers of domains) and keeps one overwritten scalar rather than a count.

Enable per run with ``NEDAS_CALL_COST=1`` in the environment, or ``CallCost(enabled=True)``.

A measured region includes any compiled work done inside it. A callback's time is therefore
an upper bound on the Python cost of that callback, not the Python cost itself -- a bound is
the useful direction, since a small one settles the question with no finer instrument.

Regions nest: the usual arrangement is one region around the whole analysis with the
callbacks measured inside it, so the parts are shares of that whole and summing every label
would double count. `report` takes the name of the enclosing region and shows the rest
against it. A label must not nest inside itself (no recursion through one region), since
each label keeps a single reusable region object.
"""
import os
import time


class _Region:
    """One reusable timed region. Not re-entrant: see the module note."""
    __slots__ = ('record', 'start')

    def __init__(self, record):
        self.record = record

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.record[0] += 1
        self.record[1] += time.perf_counter() - self.start
        return False


class _NullRegion:
    """What `measure` returns when disabled: cheaper than anything that records."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


_NULL = _NullRegion()


class CallCost:
    """
    Call counts and elapsed time per labelled region, in insertion order.

    Args:
        enabled (bool): record anything at all. Defaults to the NEDAS_CALL_COST environment
            variable, so a benchmark run can turn it on without a config change and an
            ordinary analysis pays nothing.
    """

    def __init__(self, enabled: bool = None):
        if enabled is None:
            enabled = bool(os.environ.get('NEDAS_CALL_COST'))
        self.enabled = enabled
        self.calls: dict[str, list] = {}   # label -> [count, seconds]
        self._regions: dict[str, _Region] = {}

    def _record(self, label: str) -> list:
        return self.calls.setdefault(label, [0, 0.0])

    def measure(self, label: str):
        """Time one region, e.g. a ctypes call site or a whole analysis."""
        if not self.enabled:
            return _NULL
        region = self._regions.get(label)
        if region is None:
            region = self._regions[label] = _Region(self._record(label))
        return region

    def wrap(self, func, label: str = None):
        """
        Return func with its calls counted and timed. For callbacks handed to compiled code,
        which calls them an unknown number of times. Returns func untouched when disabled, so
        the callback keeps its original call cost.
        """
        if not self.enabled:
            return func
        record = self._record(label or func.__name__)

        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                record[0] += 1
                record[1] += time.perf_counter() - start

        wrapped.__name__ = getattr(func, '__name__', 'wrapped')
        return wrapped

    def count(self, label: str) -> int:
        return self.calls.get(label, [0, 0.0])[0]

    def seconds(self, label: str) -> float:
        return self.calls.get(label, [0, 0.0])[1]

    def crossings(self, enclosing: str = None) -> int:
        """Total calls across every label but the enclosing region: boundary crossings."""
        return sum(n for label, (n, _) in self.calls.items() if label != enclosing)

    def report(self, enclosing: str = None) -> str:
        """
        A table of count, seconds and -- when `enclosing` names a measured region -- each
        label's share of it. The enclosing region is reported first, at 100%.
        """
        if not self.enabled:
            return "call cost not recorded (set NEDAS_CALL_COST=1)"
        total = self.seconds(enclosing) if enclosing else 0.0
        order = ([enclosing] if enclosing in self.calls else []) + \
                [label for label in self.calls if label != enclosing]
        width = max((len(label) for label in order), default=0)
        lines = [f"{'region'.ljust(width)}  {'calls':>9}  {'seconds':>10}  {'share':>7}"]
        for label in order:
            count, seconds = self.calls[label]
            share = f"{100 * seconds / total:6.2f}%" if total > 0 else "      -"
            lines.append(f"{label.ljust(width)}  {count:9d}  {seconds:10.4f}  {share:>7}")
        if enclosing in self.calls:
            lines.append(f"{'boundary crossings'.ljust(width)}  {self.crossings(enclosing):9d}")
        return "\n".join(lines)
