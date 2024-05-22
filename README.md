# bridger

# Don’t Start from Scratch: Behavioral Refinement via Interpolant-based Policy Diffusion
This repository contains the code for our paper [Don’t Start from Scratch: Behavioral Refinement via Interpolant-based Policy Diffusion](https://arxiv.org/pdf/2402.16075v2) (RSS-2024).

Project page: https://clear-nus.github.io/blog/bridger
## Introduction

Imitation learning empowers artificial agents to mimic behavior by learning from demonstrations. Recently, diffusion models, which have the ability to model high-dimensional
and multimodal distributions, have shown impressive performance on imitation learning tasks. These models learn to shape
a policy by diffusing actions (or states) from standard Gaussian
noise. However, the target policy to be learned is often significantly different from Gaussian and this mismatch can result
in poor performance when using a small number of diffusion steps (to improve inference speed) and under limited data. The
key idea in this work is that initiating from a more informative source than Gaussian enables diffusion methods to mitigate
the above limitations. We contribute both theoretical results, a new method, and empirical findings that show the benefits
of using an informative source policy. Our method, which we
call BRIDGER, leverages the stochastic interpolants framework to bridge arbitrary policies, thus enabling a flexible approach
towards imitation learning. It generalizes prior work in that standard Gaussians can still be applied, but other source policies
can be used if available. In experiments on challenging simulation
benchmarks and on real robots, BRIDGER outperforms state-of-the-art diffusion policies. We provide further analysis on design
considerations when applying BRIDGER.

<p align="center">
  <img src="https://github.com/clear-nus/bridger/image/bridger.png?raw=true" width="40%">
  <br />
  <span>Fig 1. verview of action generation with BRIDGER. With
trained velocity b and score s functions, BRIDGER transports the
actions from source distribution to the target distribution via the forward SDE.</span>
</p>

## Environment Setup 

The code is tested on Ubuntu 20.04, Python 3.7+ and CUDA 11.4. Please download the relevant Python packages by running:

Install Mujoco:

```
Install mujoco (https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da step3)
pip3 install mujoco-py
apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
pip3 install "cython<3"
export LD_LIBRARY_PATH=/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

```

Instal requirement.txt

```
pip3 install -r requirements.txt
```

## Usage

To train BRIDGER, run the following:
```
python train.py --task_name --model_name --beta_max --interpolant_type --prior_policy --seed --data_size
e.g. python train.py --task_name door_human --model_name si --beta_max 0.03 --interpolant_type power3 --prior_policy heuristic  --seed 1 --data_size 2500
```

To use data-drive source policy, train cvae first and then train BRIDGER

```
e.g. Please keep the seed and data_size to be the same
python train.py --task_name door_human --model_name cvae --seed 1 --data_size 2500
python train.py --task_name door_human --model_name si --prior_policy cvae --seed 1 --data_size 2500
```


To change hyperparameters, please modify ```dataset/config/[task_name]/si.json```.

## Grasp Generation Experiment
Please check https://github.com/clear-nus/bridger_grasp
## BibTeX

To cite this work, please use:

```
@article{chen2024behavioral,
  title={Don’t Start from Scratch: Behavioral Refinement via Interpolant-based Policy Diffusion},
  author={Chen, Kaiqi and Lim, Eugene and Lin, Kelvin and Chen, Yiyang and Soh, Harold},
  journal={arXiv preprint arXiv:2402.16075},
  year={2024}
}
```

### Acknowledgement 

This repo contains code that's based on the following repos: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).