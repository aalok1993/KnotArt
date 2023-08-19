# KnotArt

This repository contains the PyTorch code for 
"[Search Me Knot, Render Me Knot: Embedding Search and Differentiable Rendering of Knots in 3D](https://arxiv.org/abs/2307.08652)".

## Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Results](#results)

## Introduction

We introduce the problem of knot-based inverse perceptual art. Given multiple target images and their corresponding viewing configurations, the objective is to find a 3D knot-based tubular structure whose appearance resembles the target images when viewed from the specified viewing configurations. To solve this problem, we first design a differentiable rendering algorithm for rendering tubular knots embedded in 3D for arbitrary perspective camera configurations. Utilizing this differentiable rendering algorithm, we search over the space of knot configurations to find the ideal knot embedding. We represent the knot embeddings via homeomorphisms of the desired template knot, where the homeomorphisms are parametrized by the weights of an invertible neural network. Our approach is fully differentiable, making it possible to find the ideal 3D tubular structure for the desired perceptual art using gradient-based optimization. We propose several loss functions that impose additional physical constraints, enforcing that the tube is free of self-intersection, lies within a predefined region in space, satisfies the physical bending limits of the tube material and the material cost is within a specified budget. We demonstrate through results that our knot representation is highly expressive and gives impressive results even for challenging target images in both single view as well as multiple view constraints. Through extensive ablation study we show that each of the proposed loss function is effective in ensuring physical realizability. We construct a real world 3D-printed object to demonstrate the practical utility of our approach. To the best of our knowledge, we are the first to propose a fully differentiable optimization framework for knot-based inverse perceptual art.

## Installation

Following are the requirements:
```
torch (1.12.1)
opencv-python (4.6.0)
Pillow (9.2.0)
numpy (1.23.1)
matplotlib (3.5.3)
```

Installation commands

```
conda env create -f knotart.yml
```

OR

```
conda create -n knotart python=3.9
conda activate knotart
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install opencv-python==4.6.0.66 numpy matplotlib Pillow
```


## Usage

For running with the ellipse renderer
```
python main.py
```

For running with the capsule renderer
```
python main.py --renderer_type capsule --num_samp 100
```

## Results

<table width="100%">
  <th colspan=2><center>Optimization Evolution</center></th>
  <tr>
    <td colspan=2><img src="https://github.com/aalok1993/KnotArt/blob/main/assets/Optimization_Evolution.gif"/></td>
  </tr>
  <tr>
    <th><center>Spatiotemporal Knots Front View</center></th>
    <th><center>Spatiotemporal Knots Side View</center></th>
  </tr>
  <tr>
    <td><img src="https://github.com/aalok1993/KnotArt/blob/main/assets/Spatiotemporal_Knots_Front_View.gif"/></td>
    <td><img src="https://github.com/aalok1993/KnotArt/blob/main/assets/Spatiotemporal_Knots_Side_View.gif"/></td>
  </tr>
</table>

