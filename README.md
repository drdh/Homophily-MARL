# A Closer Look at Cooperation Emergence via Multi-Agent RL

Our method is built on [PyMARL](https://github.com/oxwhirl/pymarl) and Sequential Social Dilemma Games ([SSDG](https://github.com/eugenevinitsky/sequential_social_dilemma_games)). We use the implementation of Cleanup and Harvest Game from the SSDG repo. The implementation of our algorithm is based on independent Q-learning from the PyMARL repo.


## Installation instructions

Set up a virtual environment and install the necessary packages using `requirements.txt` file.

```bash
conda create -n SSD python=3.7
conda activate SSD
pip install -r requirements.txt
```

## Run an experiment 

```bash
python3 src/main.py 
--config=homophily 
--env-config=cleanup 
with 
env_args.map=default10
env_args.num_agents=10
use_tensorboard=True 
```

For Cleanup with 5 agents, set `env_args.map=default5` and `env_args.num_agents=5`.
For Cleanup with 3 agents, set `env_args.map=default3` and `env_args.num_agents=3`.
For Harvest, set `--env-config=harvest`.

The config files contain default settings for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`.
`--env-config` refers to the config files in `src/config/envs`.

All results will be stored in the `results` folder.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is `False` by default. The frequency of saving models can be adjusted using `save_model_interval`. Models will be saved in the result directory, under the folder called *models*. The directory corresponding to each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching replays

`save_replay` option allows saving replays of models which are loaded from `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run in the test mode. 

```bash
python3 src/main.py 
--config=homophily 
--env-config=cleanup 
with 
env_args.map=default10
env_args.num_agents=10
use_tensorboard=False 
save_replay=True 
env_args.is_replay=True 
test_nepisode=1 
checkpoint_path="results/models/model_file_name"
```

The replays can be found in `results/replays/` folder.
