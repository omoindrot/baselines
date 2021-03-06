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
cp run_frozen.sh "$LOGDIR/run_frozen.sh"


python train.py --logdir $LOGDIR\
    --game "frozen"\
    --gamma 0.99\
    --learning_rate 0.001\
    --max_timesteps 100000\
    --buffer_size 5000\
    --initial_epsilon 1.0\
    --exploration_fraction 0.2\
    --final_epsilon 0.02\
    --train_freq 1\
    --batch_size 32\
    --print_freq 10\
    --learning_starts 1000\
    --target_network_update_freq 500\
    --grad_norm_clipping 10.0\
    --eta 0.01\
    #--curiosity\
    #--visualize\
