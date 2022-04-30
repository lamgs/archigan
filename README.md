# "ArchiGAN - Artificial Architectures": PyTorch Implementation.
<!-- [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/tf-3dgan/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1610.07584)
 -->

## Introduction

* This PyTorch implementation of ArchiGAN, a research project conducted for generative form-finding based on 3D data. Code is based off of part of the [paper](https://arxiv.org/abs/1610.07584) "Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling". I provide the complete pipeline of loading dataset, training, evaluation and visualization here and also I would share some results based on different parameter settings.

### Prerequisites

* Python 3.6.5 | Anaconda
* Pytorch 0.4.1
* tensorboardX
* visdom (optional)

### Pipeline

* 3D models were converted into obj files, that were then processed using `binvox-rw.py`.

#### Data
* Data was proprietary, and needs to be placed in a folder called `volumetric_data`.

#### Training
* Then `cd src`, simply run `python main.py` on GPU or CPU. Of course, you need a GPU for training until getting good results. I used one GeForce GTX 1070 in my experiments on 3D models with resolution of 32x32x32. The maximum number of channels of feature map is 256. Because of these, the results may be inconsistent with the paper. You may need a stronger one for higher resolution one 64x64x64 and 512 feature maps. 

* Other arguments could be used, for example, `python main.py --logs=<SOMETHING_YOU_WANT_TO_LOG>` would start the tensorboardX for logging to `outputs` folder. For local debugging, you can run `python main.py --local_test=True`.

* During training, model weights and some 3D reconstruction images would be also logged to the `outputs`, `images` folders, respectively, for every `model_save_step` number of step in `params.py`. You can play with all parameters in `params.py`.

#### Evaluation
* For evaluation for trained model, you can run `python main.py --test=True` to call `tester.py`.
* If you want to visualize using visdom, first run `python -m visdom.server`, then `python main.py --test=True --use_visdom=True`.
* For more results, see the following or the `results` folder.

<!-- 
### GAN Trick
I use some more trick for better result
* the loss function to optimize G is `min (log 1-D)`, but in practice folks practically use `max log D`
* Z is Sampled from a gaussian distribution [0, 0.33]
* Use Soft Labels - It make loss function smoothing (When I don't use soft labels , I observe divergence after 500 epochs)
* learning rate scheduler - after 500 epoch, descriminator's learning rate is decayed

If you want to know more trick , 
go to  [Soumith’s ganhacks repo.](https://github.com/soumith/ganhacks)
 -->

### Basic Parameter Settings
* Here I list some basic parameter settings and in the results section I would change some specific parameters and see what happens.
* Batch size is 32, which depends on the memory and I do not see much difference by changing it.
* Learning rate, beta values for Adam and LeakyReLU parameters are the same with the original paper, as well as discriminator update trick based on accuracy.
* Latent z vector is sampled from normal(0, 0.33) following [ganhacks](https://github.com/soumith/ganhacks), but I do not use soft labels in the basic setting.
* Sigmoid function is used at both generator and discriminator for final outputs.


### Acknowledgements

* This code is a heavily modified version based on both [3DGAN-Pytorch](https://github.com/rimchang/3DGAN-Pytorch) and [tf-3dgan](https://github.com/meetshah1995/tf-3dgan) and thanks for them.


