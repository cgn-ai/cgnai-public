#!/bin/bash

# The experiment-directory this bash file is in
xdir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
finished=$XDIR/finished_tasks.txt
if [ -f "$finished" ]; then
    echo "Resume training..."
    s=$(tail -n 1 $finished)
    s=$((s+1))
else 
    echo "Start new sweep..."
    s=1
fi

# Returns the number of configurations, and
# run the experiment wrapper for each of those configurations
T=$(python $xdir/{{wrapper}} info)
for t in $(seq $s $T) 
do 
	python -B $xdir/wrapper.py run -t $t --dir $xdir/results
	echo $t >> $xdir/finished_tasks.txt 
done
