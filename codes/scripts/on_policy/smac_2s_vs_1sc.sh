export CUDA_VISIBLE_DEVICES=0
python main.py \
    --name debug \
    --trainer on_policy \
    --env smac \
    --env_args.map_name 2s_vs_1sc \
    --env_args.use_absorbing_state False \
    --use_eval False \
    --train.share_actors False \
    --train.share_critics False \
    --train.batch_length 100 \
    --critic.output symexp_twohot \
    --use_rnn True \
    --seed 1