#!/bin/bash
# log_show.sh <logfile> <from_line> <to_line>
sed -n "${2},${3}p" "$1" | sed -E 's/^\(EngineCore pid=[0-9]+\) ERROR [0-9-]+ [0-9:]+ \[core.py:[0-9]+\] //' | cut -c1-300
