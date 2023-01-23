#!/bin/bash

. $CONFIG_FILE

total_period=`diff_time $DATE_START $DATE_END`

scale=$1
i_sample=$2

cat << EOF
&perturbation

debug = .false.

xdim = $((($XEND-$XSTART)/$DX))
ydim = $((($YEND-$YSTART)/$DX))
dx = $((DX/1000))
dt = $((CYCLE_PERIOD/60))

i_sample = $i_sample
nens = $PERTURB_NUM_ENS

EOF

n_field=0
for field in ${PERTURB_VARIABLE[@]}; do
    n_field=$((n_field+1))
    echo "field($n_field)%name = '`printf '%-8s' $field`'"
    echo "field($n_field)%vars = `./param.sh $field vars $scale $i_sample`"
    echo "field($n_field)%hradius = `./param.sh $field hradius $scale $i_sample`"
    echo "field($n_field)%tradius = `./param.sh $field tradius $scale $i_sample`"
done

cat << EOF
n_field = $n_field

prsflg = 1
/
EOF
