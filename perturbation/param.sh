#!/bin/bash

. $CONFIG_FILE

varname=$1
param_type=$2
s_ind=$3
t_ind=$4
cycle_period=$(($CYCLE_PERIOD/60))

###param_type: vars, hradius, tradius
###vars is a function of lead time t_ind
###all params are a function of scale component s_ind
case $s_ind in
    1)
        hradius=1200.   ##km
        tradius=36.     ##h
        ;;
    2)
        hradius=600.
        tradius=24.
        ;;
    3)
        hradius=240.
        tradius=16.
        ;;
    4)
        hradius=120.
        tradius=12.
        ;;
    5)
        hradius=40.
        tradius=6.
        ;;
    6)
        hradius=20.
        tradius=6.
        ;;
    7)
        hradius=9.
        tradius=6.
        ;;
esac

case $varname in
    uwind|vwind)
        case $s_ind in
            1)
                vars_start=0.0  #m/s
                vars_end=3.8    #m/s
                growth_period=336 #h
                ;;
            2)
                vars_start=0.0
                vars_end=3.6
                growth_period=288
                ;;
            3)
                vars_start=0.1
                vars_end=1.3
                growth_period=240
                ;;
            4)
                vars_start=0.2
                vars_end=0.5
                growth_period=96
                ;;
            5)
                vars_start=0.15
                vars_end=0.5
                growth_period=36
                ;;
            6)
                vars_start=0.1
                vars_end=0.4
                growth_period=24
                ;;
            7)
                vars_start=0.1
                vars_end=0.4
                growth_period=12
                ;;
        esac
        period=`echo "$t_ind * $cycle_period" |bc`
        if [[ $period -gt $growth_period ]]; then
            vars=$vars_end
        else
            vars=`echo "$vars_start + ($vars_end - $vars_start) * $t_ind * $cycle_period / $growth_period" |bc -l |cut -c-5`
        fi
        ;;

    slp)
        vars=8.
        ;;
esac

##output the needed param
case $param_type in
    vars)
        echo $vars
        ;;
    hradius)
        echo $hradius
        ;;
    tradius)
        echo $tradius
        ;;
esac



