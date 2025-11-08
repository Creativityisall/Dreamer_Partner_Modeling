for seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --name train \
        --trainer dreamer \
        --env smac \
        --env_args.map_name 3s_vs_5z \
        --env_args.use_absorbing_state True \
        --env_args.trailing_absorbing_state_length 2 \
        --train.num_env_steps 400000 \
        --use_eval True \
        --replay.capacity 250000 \
        --seed $seed
    sleep 2
done
