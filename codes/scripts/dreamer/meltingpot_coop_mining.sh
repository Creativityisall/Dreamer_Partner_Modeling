for seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --name train \
        --trainer dreamer \
        --env meltingpot \
        --train.num_env_steps 1000000 \
        --use_eval True \
        --replay.capacity 250000 \
        --seed $seed
    sleep 2
done
