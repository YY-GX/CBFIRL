#!/bin/sh

#i_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
i_array=(1 2 3 4 5 6 7 8 9 10)

FOLDER="data/trpo_cbf/cbf_airl_based_on_airl_with_diff_seeds_"
i=1

cd .. && pwd &&
for i in "${i_array[@]}"; # weight
do
    mainfolder="${FOLDER}$i/share"
    python scripts/airl_safe_test.py --policy_path "${mainfolder}" --seed $i
done

