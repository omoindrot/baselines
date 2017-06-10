# Deep Q Learning + Curiosity

### Frozen Lake 8x8

- baseline: the lake is slippery
  - go up for states on the top border, hoping to go right
  - go right for states on the right border, hoping to go down
  - this gives 70.3%, 70.5%, 71.6% over 1000 runs each
  - average 70.8% of winning
  - we actually can't go in a hole with this strategy, but we can loose if it takes more than 200 iterations to get to the end
  
- standard deep Q learning
  - finds a better strategy (tries to go right at the beginning, which is risky because there are holes below)
  - average 80.3% of winning
    - sometimes dies from not finishing in time
    - sometimes dies from falling into holes (has to take risks)
  - exploration: greedy epsilon exploration
    - epsilon starts at 100%, goes down to 2% after 20,000 iterations
    - then epsilon stays at 2% for 80,000 iterations
    - first reward (first time we solved the problem): usually around 60% epsilon, i.e. ater 8,000 iterations
      - got 30% epsilon once
    - purely random exploration (only random actions) gets to the reward in 514 steps on average

#### Curiosity driven exploration
- We need to be careful when using replay memory here.
  - we need to store the intrinsic reward **at the time of the step** in the replay buffer
- Try to do importance sampling given this curiosity reward
  - Sample more episodes where the intrinsic reward was high, which should give more training information



## TODO: interesting experiments
- add optimism in Q network
  - initialize last bias to 1? to 1 / (1 - gamma) ?
- change the way we apply the intrinsic memory
  - during training vs. at play time
  - compare both results
- do we need epsilon greedy exploration with curiosity?
  - we could tend to say that we don't need it
  - but always good to have some randomness
  - without randomness the agent can get stuck in a wall forever
  - but with curiosity, can we reduce the initial epsilon / decrease it quicker ?
