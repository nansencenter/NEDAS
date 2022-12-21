#!/bin/bash

. $CONFIG_FILE

total_period=`diff_time $DATE_START $DATE_END`
num_cycle=$((total_period/$CYCLE_PERIOD))

i_sample=$1

cat << EOF
&perturbation

debug = .false.

xdim = $((($XEND-$XSTART)/$DX))
ydim = $((($YEND-$YSTART)/$DX))
dx = $((DX/1000))
dt = $((CYCLE_PERIOD/60))

n_sample = $num_cycle
i_sample = $i_sample
nens = $NUM_ENS

EOF

n_field=0
for field in ${PERTURB_VARIABLE[@]}; do
    n_field=$((n_field+1))
    param_dir=param/`padzero $i_sample 3`/$field
    echo "field($n_field)%name = '`printf '%-8s' $field`'"
    echo "field($n_field)%vars = `cat $param_dir/vars`"
    echo "field($n_field)%hradius = `cat $param_dir/hradius`"
    echo "field($n_field)%tradius = `cat $param_dir/tradius`"
done

cat << EOF
n_field = $n_field

prsflg = 1
/
EOF
