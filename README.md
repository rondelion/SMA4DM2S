# SMA4DM2S
A Sequence Memory Agent for a Delayed Match-to-Sample Task

## Features
* [BriCA](https://github.com/wbap/BriCA1) (a framework for Brain-inspired Computing Architecture)

* MinWMEnvA1.py: a simple delayed [Match-to-Sample task](https://en.wikipedia.org/wiki/Match-to-sample_task)

Details are in [this article](http://rondelionai.blogspot.com/2023/04/solving-delayed-match-to-sample-task.html).

## How to Install
* Clone the repository

* Install [BriCA](https://github.com/wbap/BriCA1)

* Register the environment to Gym
    * Place `MinWMEnvA.py` file in `gym/gym/envs/myenv`  
    (wherever Gym to be used is installed)
    * Add to `__init__.py` (located in the same folder)  
      `from gym.envs.myenv.MinWMEnvA1 import MinWMEnvA1`
    * Add to `gym/gym/envs/__init__.py`  
```
register(
    id='MinWMEnvA1-v0',
    entry_point='gym.envs.myenv:MinWMEnvA1'
    )
```

## Usage
### Command arguments
- Options
      --dump: dump file path
      --episode_count: Number of training episodes (default: 1)
      --max_steps: Max steps in an episode (default: 50)
      --config: Model configuration (default: SMA4DM2S.json)
      --dump_level >= 0

### Sample usage
```
$ python SMA4DM2S.py --config SMA4DM2S.2.2.json --episode_count 1000 --dump "dump.txt" --dump_level 2

```

## Other files

* SMA4DM2S.2.2.json, SMA4DM2S.3.3.json:	config. files  

* SMA4DM2S.brical.json  brical file

* MinM2SRL.py an RL agent for comarison

* MinM2SRL.json config file for MinM2SRL.py


