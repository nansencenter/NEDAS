#!/bin/bash
function advance_time {
  ##input time format ccyymmddHHMM
  ccyymmdd=`echo $1 |cut -c1-8`
  hh=`echo $1 |cut -c9-10`
  mm=`echo $1 |cut -c11-12`

  ##input time delta
  inc=$2  ##increment in hours
  inc_sec=`bc <<< "($inc * 3600 + 0.5)/1"`  ##convert to seconds

  ##which date command to use (on MacOS use gdate)
  if command -v gdate &> /dev/null; then
      date=gdate
  else
      date=date
  fi

  ##make the date calculation
  $date -u -d "$inc_sec seconds $ccyymmdd $hh:$mm" +%Y%m%d%H%M
}
export -f advance_time


function diff_time {
    date1=$1
    date2=$2
    d=0
    while [[ $((date1/10000)) -ne $((date2/10000)) ]]; do
        if [[ $date1 -lt $date2 ]]; then
            d=$((d+1440))
            date1=`advance_time $date1 1440`
        else
            d=$((d-1440))
            date1=`advance_time $date1 -1440`
        fi
    done
    while [[ $((date1/100)) -ne $((date2/100)) ]]; do
        if [[ $date1 -lt $date2 ]]; then
            d=$((d+60))
            date1=`advance_time $date1 60`
        else
            d=$((d-60))
            date1=`advance_time $date1 -60`
        fi
    done
    while [[ $date1 -ne $date2 ]]; do
        if [[ $date1 -lt $date2 ]]; then
            d=$((d+1))
            date1=`advance_time $date1 1`
        else
            d=$((d-1))
            date1=`advance_time $date1 -1`
        fi
    done
    echo $d
}
export -f diff_time


function format_time_string {
    ccyy=`echo $1 |cut -c1-4`
    mm=`echo $1 |cut -c5-6`
    dd=`echo $1 |cut -c7-8`
    hh=`echo $1 |cut -c9-10`
    ii=`echo $1 |cut -c11-12`
    echo ${ccyy}-${mm}-${dd}_${hh}:${ii}:00
}
export -f format_time_string


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
    l=`cat $logfile |wc -m`
    t=0
    until [ -s $logfile ]; do sleep 10 ; done
    until [[ `tail -n5 $logfile |grep "$keyword"` ]]; do
        sleep 60
        l1=`cat $logfile |wc -m`
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
        sleep 60
        t=$((t+1))
        if [ $t -gt $timeout ]; then
            echo Timeout waiting for $filename. Abort.
            echo error > $rundir/stat
            exit 1
        fi
    done
}
export -f watch_file


function padzero {
    str=$1
    len=$2
    expr $1 + $((10**$len)) |cut -c2-
}
export -f padzero


# Function to clean up and terminate all child processes
function cleanup {
    echo "cleaning up and terminating..."
    pkill -P $$
    exit
}
export -f cleanup

