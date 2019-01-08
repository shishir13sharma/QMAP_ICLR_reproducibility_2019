Q-map: a Convolutional Approach for Goal-Oriented Reinforcement Learning [https://openreview.net/forum?id=rye7XnRqFm]

Installation
First make sure you have Pytorch, Baselines, Gym and Gym Retro installed. This code was written for versions 0.4.1, 0.1.5, 0.10.5 and 0.6.0 of these libraries.

To install this package, run:

git clone https://github.com/fabiopardo/qmap.git
cd qmap
pip install -e .
and copy the SuperMarioAllStars-Snes folder to the retro/data/stable directory where Gym Retro is installed.
Finally clone this repository into the qmap folder

Usage: python train_mario.py

For loading a previously saved model, simply pass the DQN or Q-Map files step value as load argument.
The training of the agent can also be accomplished by running the jupyter notebook train_mario.ipynb