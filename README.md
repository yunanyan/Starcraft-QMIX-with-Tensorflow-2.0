# Starcraft-QMIX-with-Tensorflow-2.0
Acknowledgement

Based on https://github.com/starry-sky6688/StarCraft. Migrated QMIX networks from Pytorch to Tensorflow 2.0. 

## Starcraft
Tensorflow 2.0 implementations of the multi-agent reinforcement learning algorithms, including 
[QMIX](https://arxiv.org/abs/1803.11485)

## Corresponding Papers
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)

## Requirements
- python
- TensorFlow 2.0
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)

## Quick Start

```shell
$ python main.py --map=3m --alg=qmix
```
