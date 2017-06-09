python train_frozen.py \
    --gamma 0.99 \
    --learning_rate 0.001 \
    --max_timesteps 50000 \
    --replay_memory \
    --buffer_size 20000 \
    --initial_epsilon 1.0 \
    --exploration_fraction 0.3 \
    --final_epsilon 0.02 \
    --train_freq 5 \
    --batch_size 32 \
    --print_freq 10 \
    --learning_starts 1000 \
    --target_network_update_freq 500 \
    #--grad_norm_clipping 10.0 \  # not ready yet
    #--visualize \
    --curiosity \
    --hiden_phi 16 \
    --eta 0.01 \
