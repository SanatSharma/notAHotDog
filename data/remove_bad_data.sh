#!/bin/bash
scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
base_dir=$(dirname "$0")
raw_data_dir="$base_dir"
echo "base dir " $raw_data_dir
declare -a class_names=(
	"neutral"
    "amateur"
    "penis_large"
	)

for cname in "${class_names[@]}"
do
	bad_images_file="$raw_data_dir/$cname/bad_$cname.txt"
	urls_file="$raw_data_dir/$cname/urls_$cname.txt"
    echo $images_dir
    while read p; do
        sed -i "$p/d" $urls_file
    done <$bad_images_file
done