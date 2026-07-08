import traceback
from typing import Any
from NEDAS.schemes.filter import FilterAnalysisScheme
from NEDAS.utils.parallel import abort_all_ranks

class ForecastScheme(FilterAnalysisScheme):
    """
    Forecast scheme class.

    This scheme runs only the ensemble forecasts for the start times defined by time_start time_end every cycle_period.
    The length of each forecast between cycles is forecast_period, which can be different from cycle_period.
    """

    def run_all(self) -> None:

        self.c.log_event("START FORECASTS...")
        while self.c.time < self.config.time_end:
            self.c.log_event(f"CURRENT START TIME: {self.c.time}")

            if self.config.run_preproc:
                self.run_step('preprocess')
                if self.config.perturb:
                    self.run_step('perturb')

            # advance model state to next analysis cycle
            if self.config.run_forecast:
                self.run_step('ensemble_forecast')

            # compute diagnostics
            if self.config.run_diagnose:
                if self.config.diag:
                    self.run_step('diagnose')

            # advance to next cycle
            self.c.time = self.c.next_time

        self.c.log_event("ALL FINISHED.", flag='finish')

    def get_task_opts(self, step:str, model_name:str, **other_opts) -> dict[str, Any]:
        opts = super().get_task_opts(step, model_name, **other_opts)
        opts['forecast_period'] = self.config.forecast_period
        return opts

def main():
    scheme = None
    try:
        # initialize scheme
        scheme = ForecastScheme(parse_args=True)

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
