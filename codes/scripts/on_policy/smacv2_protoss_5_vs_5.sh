export CUDA_VISIBLE_DEVICES=7
python main.py \
    --name benchmark \
    --trainer on_policy \
    --env smacv2 \
    --env_args.map_name protoss_5_vs_5 \
    --env_args.use_absorbing_state False \
    --use_eval True \
    --train.share_actors True \
    --train.share_critics True \
    --train.batch_length 400 \
    --critic.output symexp_twohot \
    --use_rnn True \
    --seed 0
