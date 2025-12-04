directories=../collected_demos

for dir in "$directories"/*; do
    dir=$(basename ${dir})
    echo "$dir"
    python3 move_cube.py --env $dir -d $directories --rollout_num 20 -r
done
