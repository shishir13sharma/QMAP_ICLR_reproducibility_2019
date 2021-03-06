
# Reproducibility Challenge: Q-map: a Convolutional Approach for Goal-Oriented Reinforcement Learning 

This repository pertains to the ICLR 2019 reproducibility challenge submission for the paper Q-map: a Convolutional Approach for Goal-Oriented Reinforcement Learning. [https://openreview.net/forum?id=rye7XnRqFm]

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. 

### Prerequisites

```
Pytorch (0.4.1)
Baselines (0.1.5)
Gym (0.10.5)
Gym Retro (0.6.0)

```

### Installing

```
git clone https://github.com/fabiopardo/qmap.git
cd qmap  
pip install -e .  
```
Copy the SuperMarioAllStars-Snes folder to the retro/data/stable directory where Gym Retro is installed.  
Finally clone this repository into the qmap folder  \

<!-- End with an example of getting some data out of the system or using it for a little demo -->

## Running

```
python train_mario.py  
```

For loading a previously saved model, simply pass the DQN or Q-Map files step value as load argument.  
The training of the agent can also be accomplished by running the jupyter notebook train_mario.ipynb  


## Authors

* **Shishir Sharma** - (https://github.com/shishir13sharma)

## Acknowledgments

* Code reproduced from https://github.com/fabiopardo/qmap
