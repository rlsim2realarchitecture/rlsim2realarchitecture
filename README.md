# Exploring Meta Learning Architectures for Sim2real Deep Reinforcement Learning with Dynamics Randomization

This repository contains source code for the experiment of "Exploring Meta Learning Architectures for Sim2real Deep Reinforcement Learning with Dynamics Randomization".

This is a [youtube link](https://youtu.be/dfMVbKDD2R0) for our experiments.

## Hyperparameters for experiment

### Hyperparameters of network architectures
We conducted bayesian optimization for hyperparameters of network architectures.

MLP's hidden size: 647  
LSTM's hidden size: 734, cell size: 70  
CNP's hidden size: 163, r size: 83  
SNAIL's number of channels: 110, number of keys: 27, number of temporal convolution filter: 8  

### Hyperparameters of PPO

We used PPO for all experiment, and fixed hyperparameters for PPO.

policy learning rate: 3e-4  
value function learning rate: 3e-4  
clip parameter: 0.2  
max grad norm: 0.5  
number of epoch per iteration: 10  
batch size: 8192  


## example command to run
hopper environment with cuda option
### mlp1
```
cd script
python run_ppo_mlp.py --timestep 1 --cuda 0 --env_name hopper_random --easy_hard hard
```
### mlp64
```
cd script
python run_ppo_mlp.py --cuda 0 --env_name hopper_random --easy_hard hard
```

### lstm
```
cd script
python run_ppo_lstm.py --cuda 0 --env_name hopper_random --easy_hard hard
```

### cnp
```
cd script
python run_ppo_cnp.py --cuda 0 --env_name hopper_random --easy_hard hard
```

### cnppe
```
cd script
python run_ppo_cnp.py --use_pe --cuda 0 --env_name hopper_random --easy_hard hard
```

### snail
```
cd script
python run_ppo_snail.py --cuda 0 --env_name hopper_random --easy_hard hard
```

you can run other environments with `--env_name torque_pendulum` and `--env_name torque_pendulum2`
