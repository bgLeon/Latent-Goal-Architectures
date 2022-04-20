# Relational Deep Reinforcement Learning and Latent Goals for Following Instructions in Temporal Logic
PLEASE, DO NOT DISTRIBUTE
This code is still in state of internal use and needs of further cleaning and documentation. An updated and cleaned version will be released in the upcoming months

## Installation instructions

This repository requires [Python3.6](https://www.python.org/) with three libraries: [numpy](http://www.numpy.org/), [tensorflow (V2.3)](https://www.tensorflow.org/), and [gym](https://gym.openai.com/). 

## Running example

    python run_experiment.py --visual --pretrained_encoder --mapType=minecraft-insp --syntax=TTL2 --num_layers=1 --network=PrediNet --mem --multimodal --num_neurons=50 --tback=1 --seedB=0 --batch_size=512  --mem_type=rim --loss=cce  (PrediNet)

This runs the BRIM$^{LG}$  from the main document, (once a pretrain encoder is available)

run_experiment.py contains instructions for each command


Note that to switch between the various experiments present in the main paper, manual changes should be done in the code.
We are also working to automatize all the switches for the final version of this code.

