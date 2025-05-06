#!/bin/bash

python -m NEDAS.models.qg.generate_truth -c /app/config.yml

python -m NEDAS.models.qg.generate_init_ensemble -c /app/config.yml
