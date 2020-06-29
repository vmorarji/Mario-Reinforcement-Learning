# Mario-Reinforcement-Learning

#### Using Q-learning to play Super Mario Bros for the Nes.

Both neural networks and replay buffers are based on tutorials from [Adventures in Machine Learning](https://adventuresinmachinelearning.com/)



### Requirements

gym:		    `pip install gym`

nes-py:		  `pip install nes-py`

Super Mario:`pip install gym-super-mario-bros`

numpy:      `pip install numpy`

tensorflow: `pip install tensorflow`
[visual studio](https://visualstudio.microsoft.com/):	For windows install the "Desktop development with C++" workload


### Training the Agent

`python main.py` 

|Arg|Default|Description
|------------------|-------------------------------|-----------------------------|
|`--env`|SuperMarioBros-1-1-v0|Gym environment and level|
|`--dueling`||Change from DDQN to Dueling DQN
|`--per`||Change from stander replay buffer to Performance Experience Replay|
|`--frame_size`|(84, 84)| Change the Resolution to decrease computational load|
|`--max_timesteps`|5e5|Total timesteps the environment will play|
|`--delay_timesteps`|1e4|Initial timesteps to fill the replay buffer|
|`--min_epsilon`|0.01|The lowest value for epsilon|
|`--epsilon_decay`|45e4|Timesteps required to reach the minimum epsilon|
|`--beta_decay`|45e4|Timesteps required to reach the maximum beta|
|`--learning_rate`|1e-3|Learning rate for the optimizer|
|`--gamma`|0.99|The discount factor for Q(s',a')|
|`--tau`|0.05|The rate for updating the target network|
|`--memory_size`|131072|The size of the memory buffer. PER requires 2^x')|
|`--batch_size`|64|The size of the mini-batches for Q learning|
|`--debug_summary`||Save tensorboard summaries of training|
|`--save_model`||Store the models at set intervals|
|`--save_freq`|5e4|The intervals when the models' weights are saved|

For Example, if you wanted to train a DDQN Model with a PER memory buffer using mini-batches of 32 samples:

```python 
python main.py --per --batch_size 32
```
