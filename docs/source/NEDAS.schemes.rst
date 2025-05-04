Analysis schemes
================



Parallelization strategy
------------------------
A compromise is made in favor of code flexibility than its runtime efficiency. We aim for more modular design so that components in the DA algorithm can be easily changed/upgraded/compared. A pause-restart strategy is used: the model writes the state to restart files, then DA reads those files and computes the analysis and outputs to the updated files, and the model continues running. This is "offline" assimilation. In operational systems, sometimes we need "online" algorithms where everything is hold in the memory to avoid slow file I/O.  NEDAS provides parallel file I/O, not suitable for time-critical applications, but efficient enough for most research and development purposes.

The first challenge on dimensionality demands a careful design of memory layout among processors. The ensemble model state has dimensions: `member`, `variable`, `time`, `z`, `y`, `x`. When preparing the state, it is easier for a processor to obtain all the state variables for one member, since they are typically stored in the same model restart file. Each processor can hold a subset of the ensemble states, this memory layout is called "state-complete". To apply the ensemble DA algorithms, we need to transpose the memory layout to "ensemble-complete", where each processor holds the entire ensemble but only for part of the state variables (`Anderson & Collins 2007 <https://doi.org/10.1175/JTECH2049.1>`_).

In NEDAS, for each member the model state is further divided into "fields" with dimensions (`y`, `x`) and "records" with dimensions (`variable`, `time`, `z`). Because, as the model dimension grows, even the entire state for one member maybe too big for one processor to hold in its memory. The smallest unit is now the 2D field, and each processor holds only a subset along the record dimension. Accordingly, the processors (`pid`) are divided into "member groups" (with same `pid_rec`) and "record groups" (with same `pid_mem`), see Fig. 1 for example. "State-complete" now becomes "field-complete". The record dimension allows parallel processing of different fields by the `read_var` functions in model modules. And during assimilation, each `pid_rec` only solves the analysis for its own list of `rec_id`.


figure ../imgs/transpose.png

Transpose from field-complete to ensemble-complete, illustrated by a 18-processor memory layout (`pid` = 1, ..., 18), divided into 2 groups (`pid_rec` = 0, 1), each with 9 processors (`pid_mem` = 0, 1, 2). The data has dimensions `mem_id` = 1:100 members, `par_id` = 1:9 partitions, and `rec_id` = 1:4 records. The gray arrows show sending/receiving of data to perform the transpose. The yellow arrows is an additional collection step (only needed by observation data)

For observations, it is easier to process the entire observing network at once, instead of going through the measurements one by one. Therefore, each observing network (record) is assigned a unique `obs_rec_id` to be handled by one processor.
Each `pid_rec` only needs to process its own list of `obs_rec_id`. Processors with `pid_mem` = 0 is responsible for reading and processing the actual observations using `read_obs` functions from dataset modules, while all `pid_mem` separately process their own members for the observation priors.
When transposing from field-complete to ensemble-complete is done, an additional collection step among different `pid_rec` is required, which gathers all `obs_rec_id` for each `rec_id` to form the final local observation.

When the transpose is complete, on each `pid`, the local ensemble state\_prior[`mem_id`, `rec_id`][`par_id`] is updated to the posterior state\_post, using local observations lobs[`obs_rec_id`][`par_id`] and observation priors lobs\_prior[`mem_id`, `obs_rec_id`][`par_id`].

.. image:: ../imgs/memory_layout.png
   :width: 100%
   :align: center

Memory layout for state variables (square pixels) and observations (circles) using (a) batch and (b) serial assimilation strategies. The colors represent the processor id `pid_mem` that stores the data.

NEDAS.schemes.get\_analysis\_scheme module
------------------------------------------

.. automodule:: NEDAS.schemes.get_analysis_scheme
   :members:
   :show-inheritance:
   :undoc-members:

NEDAS.schemes.base module
-------------------------

.. automodule:: NEDAS.schemes.base
   :members:
   :show-inheritance:
   :undoc-members:

NEDAS.schemes.offline\_filter module
------------------------------------

.. automodule:: NEDAS.schemes.offline_filter
   :members:
   :show-inheritance:
   :undoc-members:
