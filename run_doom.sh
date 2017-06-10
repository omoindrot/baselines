LOGDIR="$1"  # take first argument for logdir

# Check that LOGDIR doesn't exist yet
if [ -d "$LOGDIR" ]; then
  echo "LOGDIR already exists: $LOGDIR"
  while true; do
    read -p "Do you wish to remove this directory and create a new one? (Y/N)" yn
    case $yn in
      [Yy]* ) rm -rf $LOGDIR; break;;
      [Nn]* ) exit;;
      * ) echo "Please answer yes or no.";;
    esac
  done
fi

echo "Logging in $LOGDIR"
mkdir $LOGDIR
cp run_doom.sh "$LOGDIR/run_doom.sh"


python train.py --logdir $LOGDIR\
    --game "doom"\
    --gamma 0.99\
    --learning_rate 0.0001\
    --max_timesteps 5000000\
    --buffer_size 50000\
    --initial_epsilon 1.0\
    --exploration_fraction 0.1\
    --final_epsilon 0.01\
    --train_freq 4\
    --batch_size 32\
    --print_freq 1\
    --learning_starts 10000\
    --target_network_update_freq 1000\
    --grad_norm_clipping 10.0\
    --eta 0.01\
    #--curiosity\
    #--visualize\
