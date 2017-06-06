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

- curiosity driven exploration
  - 


