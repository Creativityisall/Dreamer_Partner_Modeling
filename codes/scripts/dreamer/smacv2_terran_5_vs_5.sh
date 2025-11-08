for seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --name train \
        --trainer dreamer \
        --env smacv2 \
        --env_args.map_name terran_5_vs_5 \
        --env_args.use_absorbing_state True \
        --env_args.trailing_absorbing_state_length 2 \
        --train.num_env_steps 500000 \
        --use_eval True \
        --replay.capacity 250000 \
        --seed $seed
    sleep 2
done
