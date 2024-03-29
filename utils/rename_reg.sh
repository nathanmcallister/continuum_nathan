#!/bin/bash

date_str=$(date '+%m_%d_%y')
date_count=0
reg_dir="regs"

for file in ./$reg_dir/reg_$date_str*; do
	[ -e "$file" ] || continue
	echo "Existing file: $file"
	((date_count++))
done

output_file="$reg_dir/reg_$date_str$(printf "\\$(printf '%03o' $((97+$date_count)))").csv"
mv output.csv "$output_file"
echo "Output file: $output_file"
