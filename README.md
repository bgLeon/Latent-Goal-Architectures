# Latent-Goal-Architectures

Source code of the paper `In a Nutshell, the Human Asked for This: Latent Goals for Following Temporal Specifications` @ `ICLR 2022` and the early preprint version titled as "Relational Deep Reinforcement Learning and Latent Goals for Following Instructions in Temporal Logic" @ ICML 2021 Workshop on Uncertainty and Robustness in Deep Learning.

## Installation instructions

This repository requires [Python3.6](https://www.python.org/) with three libraries: [numpy](http://www.numpy.org/), [tensorflow (V2.5)](https://www.tensorflow.org/), and [gym](https://gym.openai.com/). 

## Running example

    python run_experiment.py --visual --pretrained_encoder --mapType=minecraft-insp --syntax=TTL2 --num_layers=1 --network=PrediNet --mem --multimodal --num_neurons=50 --tback=1 --seedB=0 --batch_size=512  --mem_type=rim --loss=cce

This runs the BRIM$^{LG}$ from the main document in the minecraft-inspired benchmark, (once a pretrain encoder is available), remove "--pretrained_encoder" if you want to train the agents in one go (note that it will take longer to converge, specially in the Minigrid setting and in the larger maps).

run_experiment.py contains instructions for each command


