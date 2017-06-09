python train_cartpole.py \
    --gamma 1.0 \
    --learning_rate 0.001 \
    --max_timesteps 100000 \
    --replay_memory \
    --buffer_size 50000 \
    --initial_epsilon 1.0 \
    --exploration_fraction 0.1 \
    --final_epsilon 0.02 \
    --train_freq 1 \
    --batch_size 32 \
    --print_freq 10 \
    --learning_starts 1000 \
    --target_network_update_freq 500 \
    #--grad_norm_clipping 10.0 \  # not ready yet
    #--visualize \
    #--curiosity \
    #--hiden_phi 16 \
    #--eta 0.01 \
