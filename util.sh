#!/bin/bash
function advance_time {
  ccyymmdd=`echo $1 |cut -c1-8`
  hh=`echo $1 |cut -c9-10`
  mm=`echo $1 |cut -c11-12`
  inc=$2
  date -u -d $inc' minutes '$ccyymmdd' '$hh':'$mm +%Y%m%d%H%M
}
export -f advance_time

function wait_for_module {
  for module in $*; do
    until [ -f $module/stat ]; do sleep 10; done
    until [[ `cat $module/stat` == "complete" ]]; do 
      sleep 15
      if [[ `cat $module/stat` == "error" ]]; then
        exit 1
      fi
    done
  done
}
export -f wait_for_module

function watch_log {
  logfile=$1
  keyword=$2
  timeout=$3
  rundir=$4
  l=0
  t=0
  until [ -s $logfile ]; do sleep 10 ; done
  until [[ `tail -n5 $logfile |grep $keyword` ]]; do
    sleep 1m
    l1=`cat $logfile |wc -l`
    if [ $l1 -eq $l ]; then
      t=$((t+1))
    else
      l=$l1
      t=0
    fi
    if [ $t -gt $timeout ]; then
      echo "`pwd`/$logfile stagnant for $timeout minutes! Abort."
      echo error > $rundir/stat
      exit 1
    fi
  done
}
export -f watch_log

function watch_file {
  filename=$1
  timeout=$2
  rundir=$3
  t=0
  until [ -f $filename ]; do
    sleep 1m
    t=$((t+1))
    if [ $t -gt $timeout ]; then
      echo Timeout waiting for $filename. Abort.
      echo error > $rundir/stat
      exit 1
    fi
  done
}
export -f watch_file
