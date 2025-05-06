#!/bin/bash

python -m NEDAS.models.qg.generate_truth -c /work/config.yml

python -m NEDAS.models.qg.generate_init_ensemble -c /work/config.yml
