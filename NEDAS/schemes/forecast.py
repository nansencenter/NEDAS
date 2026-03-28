from typing import Any
from NEDAS.schemes.filter import FilterAnalysisScheme

class ForecastScheme(FilterAnalysisScheme):
    """
    Forecast scheme class.

    This scheme runs only the ensemble forecasts for the start times defined by time_start time_end every cycle_period.
    The length of each forecast between cycles is forecast_period, which can be different from cycle_period.
    """

    def run_all(self) -> None:

        self.c.print_1p("START FORECASTS...\n")
        while self.c.time < self.config.time_end:
            msg = f"CURRENT START TIME: {self.c.time}"
            self.c.print_1p(f"{msg}\n{'='*len(msg)}\n")

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

        self.c.print_1p("ALL FINISHED.\n")

    def get_task_opts(self, model_name:str, **other_opts) -> dict[str, Any]:
        opts = {
            'model_src': model_name,
            'time': self.c.time,
            'forecast_period': self.config.forecast_period,
            **(self.config.job_submit or {}),
            **other_opts,
        }
        return opts

def main():
    # initialize scheme
    scheme = ForecastScheme(parse_args=True)

    step = scheme.config.step
    if step:
        scheme.run_step(step)
        return

    scheme()

if __name__ == '__main__':
    main()
