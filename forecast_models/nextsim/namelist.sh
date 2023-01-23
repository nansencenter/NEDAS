#!/bin/bash

. $CONFIG_FILE

cat << EOF
[mesh]
filename=small_arctic_10km.msh

[setup]
ice-type=topaz
ocean-type=topaz
atmosphere-type=generic_ps
bathymetry-type=etopo
use_assimilation=false
dynamics-type=bbm

[simul]
timestep=900
time_init=${DATE:0:4}-${DATE:4:2}-${DATE:6:2} ${DATE:8:2}:00:00
duration=21

[thermo]
use_assim_flux=false
diffusivity_sss=0
diffusivity_sst=0
ocean_nudge_timeS=1296000
ocean_nudge_timeT=1296000
h_young_max=0.23
alb_ice=0.69
alb_sn=0.88
albedoW=0.07

[dynamics]
time_relaxation_damage=15
compression_factor=10e3
C_lab=1.35e6
substeps=120
use_temperature_dependent_healing=true
ERA5_quad_drag_coef_air=0.0017
quad_drag_coef_water=0.0061625
Lemieux_basal_k1=7
compaction_param=-25

[restart]
start_from_restart=true
basename=${DATE:0:8}T${DATE:8:2}0000Z
input_path=restart
type=extend
write_initial_restart=true
write_interval_restart=true
output_interval=1
check_restart=true

[output]
output_per_day=`expr 1440 / $CYCLE_PERIOD`
datetime_in_filename=true
exporter_path=.
export_before_regrid=true
export_after_regrid=true

[debugging]
check_fields_fast=false

EOF
