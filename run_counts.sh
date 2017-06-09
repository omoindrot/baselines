python count_based_frozen.py \
    --gamma 0.99 \
    --learning_rate 0.001 \
    --max_timesteps 50000 \
    --buffer_size 20000 \
    --initial_epsilon 0.5 \
    --exploration_fraction 0.3 \
    --final_epsilon 0.02 \
    --train_freq 5 \
    --batch_size 1 \
    --print_freq 10 \
    --learning_starts 0 \
    --target_network_update_freq 500 \
    --visualize \
    --eta 0.01 \
    #--replay_memory \
    #--grad_norm_clipping 10.0 \  # not ready yet
