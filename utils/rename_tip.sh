#!/bin/bash

date_str=$(date '+%m_%d_%y')
date_count=0
tip_dir="tip_cals"

for file in ./$tip_dir/tip_cal_$date_str*; do
	[ -e "$file" ] || continue
	echo "Existing file: $file"
	((date_count++))
done

output_file="$tip_dir/tip_cal_$date_str$(printf "\\$(printf '%03o' $((97+$date_count)))").csv"
mv output.csv "$output_file"
echo "Output file: $output_file"
