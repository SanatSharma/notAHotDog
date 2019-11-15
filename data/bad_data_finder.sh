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
	urls_file="$raw_data_dir/$cname/urls_$cname.txt"
	images_dir="$raw_data_dir/$cname/IMAGES"
    echo $images_dir
    duplicate_file="$raw_data_dir/$cname/bad_$cname.txt"
    test="s.txt"
    echo $duplicate_file
    echo $temp
    cd $images_dir
    find . -name "*.jpg" -type 'f' -size -504c > $duplicate_file
    sed 's/^.//' $duplicate_file > $test
    mv $test $duplicate_file
    echo "Class: $cname. Total duplicates found : $(cat $duplicate_file | wc -l)"
    
done