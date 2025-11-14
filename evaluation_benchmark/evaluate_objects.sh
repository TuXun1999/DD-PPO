directories=../collected_models

for dir in "$directories"/*; do
    dir=$(basename ${dir})
    echo "$dir"
    python3 move_cube.py --env $dir --rollout_num 20 -r
done
