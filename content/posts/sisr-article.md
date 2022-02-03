+++ 
date = 2022-02-02T21:38:54+05:45
title = "Single Image Super Resolution"
description = ""
slug = ""
authors = ["Sulav Timilsina","Milan Gautam"]
tags = []
categories = ["Deep Learning"]
externalLink = ""
series = ["Artificial Intelligence"]
+++

## Introduction
[GAN](https://arxiv.org/abs/1406.2661) (Generating Adversarial Network) is about creating, like drawing but completely from scratch. It has got two models: the Generator and the Discriminator are put together into a game of adversary. Through which they learn the intricate details of the target data distribution. These two models are playing a MIN-MAX game where one tries to minimize the loss and the other tries to maximize. While doing so a global optimum is reached, where the Discriminator is no longer able to distinguish between real and generated (fake) data distribution. 

SISR(Single Image Super-Resolution) is an application of GAN. Image super-resolution is the process of enlarging small photos while maintaining a high level of quality, or of restoring high-resolution images from low-resolution photographs with rich information. Here the model's work is to map the function from low-resolution image data to its high-resolution image. Instead of giving a random noise to the Generator, a low-resolution image is fed into it. After passing through various Convolutional Layers and Upsampling Layers, the Generator gives a high-resolution image output. Generally, there are multiple solutions to this problem, so it's quite difficult to master the output up to original images in terms of richness and quality.

## Data Preprocessing and Augmentation:

We have used the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) [Agustsson and Timofte (2017)] dataset provided by the TensorFlow library. There are altogether 800 pairs of low resolution and high-resolution images in the the training set whereas 100 pairs in the testing set. This data contains mainly people, cities, fauna, sceneries, etc.
We used flipping and rotating through 90, 180, and 270 degrees randomly over the dataset. Since we had limited memory on the training computer, we had to split large images into patches of smaller size. 19 patches of size 96 ✕ 96 pixels resolution were obtained from an image randomly. Our generator is designed to upsample images by 4 times so, the output image patch will be of dimension: 384 ✕ 384 pixels. 

## Generator

The generator is the block in the architecture which is responsible for generating the high resolution(HR) images from low resolution(LR) images. In 2015, [SRGAN](https://arxiv.org/abs/1609.04802) was published which introduced the concept of using GAN for SISR tasks which produced the state the art solution. The generator of [SRGAN](https://arxiv.org/abs/1609.04802) consists of several residual blocks that facilitate the flow of the gradient during backpropagation. 

![Generator Architecture](/images/generator.png#center)


To further enhance the quality of generator images [ESRGAN](https://arxiv.org/abs/1809.00219) was released which performed some modifications in the generator of the [SRGAN](https://arxiv.org/abs/1609.04802) which includes:
- Removing the batch normalized(BN) layers.
- Replacing the original residual block with the proposed Residual-in-Residual Dense Block (RRDB), which combines multi-level residual network and dense connections as in the figure below:

![RRDB Diagram](/images/rrdb.png#center)

Fig: Residual in Residual Dense Block(RRDB) 

Removing BN layers has proven to increase performance and reduce the computational complexity in different PSNR-oriented tasks including SR  and deblurring. BN layers normalize the features using mean and variance in a batch during training and use the estimated mean and variance of the whole training dataset during testing. When the statistics of training and testing datasets differ a lot, BN layers tend to introduce unpleasant artifacts and limit the generalization ability. The researchers empirically observe that BN layers are more likely to bring artifacts when the network is deeper and trained under a GAN framework. These artifacts occasionally appear among iterations and different settings, violating the need for stable performance overtraining.  Therefore, removing BN layers for stable training and consistent performance. Furthermore, removing BN layers helps to improve generalization ability and to reduce computational complexity and memory usage.

RRDB employs a deeper and more complex structure than the original residual block in SRGAN. Specifically, as shown in the figure above, the proposed RRDB has a residual-in-residual structure, where residual learning is used at different levels. Here, the RRDB uses dense block in the main path, where the network capacity becomes higher benefiting from the dense connections. In addition to the improved architecture, it also exploits several techniques to facilitate training a very deep network such as residual scaling(beta) i.e., scaling down the residuals by multiplying a constant between 0 and 1 before adding them to the main path to prevent instability.

## Discriminator

The task of the discriminator is to discriminate between real HR images and generated SR images. Discriminator architecture here used is similar to DC-GAN architecture with LeakyReLU activation function. The network contains eight convolutional layers with 3×3 filter kernels, increasing by a factor of 2 from 64 to 512 kernels as in the VGG network. Strided convolutions are used to reduce the image resolution each time the number of features is doubled. 
But to overcome the instability while training of original GAN, we use a variant of GANs named improved training of Wasserstein GANs (WGAN-GP). So the last sigmoid layer of the conventional DC-GAN discriminator is omitted. This helps in not restricting the feature maps in 0 to 1 value.

![discriminator image](/images/discriminator.png#center)

## Losses: 

### Generator Loss

The generator loss is the sum of MSE, perceptual loss +adversarial loss

_l<sub>G</sub> = MSE+Perceptual Loss +Adversarial loss_

_l<sub>G</sub>= l<sub>MSE</sub>+l<sub>p</sub>+ l<sub>GA</sub>_

### Mean Square Error(MSE)

As the most common optimization objective for SISR, the pixelwise MSE loss is calculated as:

_l<sub>MSE</sub> = ||G<sub>Θ</sub>(I<sub>LR</sub>) - I<sub>HR</sub>||<sub>2</sub><sup>2</sup>_,

where the parameter of the generator is denoted by ; the generated image, namely I<sub>SR</sub>,is denoted by G<sub>Θ</sub>(I<sub>LR</sub>); and the ground truth is denoted by I<sub>HR</sub> . Although models with MSE loss favor a high PSNR value, the generated results tend to be perceptually unsatisfying with overly smooth textures. Despite the aforementioned shortcomings, this loss term is still kept because MSE has clear physical meaning and helps to maintain color stability.


# mse code here

### Perceptual Loss
To compensate for the shortcomings of MSE loss and allow the loss function to better measure semantic and perceptual differences between images, we define and optimize a perceptual loss based on high-level features extracted from a pretrained network. The rationality of this loss term lies in that the pretrained network for classification originally has learned to encode the semantic and perceptual information that may be measured in the loss function. To enhance the performance of the perceptual loss, a 19-layer VGG network is used. The perceptual loss is actually the Euclidean distance between feature representations, which is defined as

_l<sub>p</sub> = ||𝜙(G<sub>Θ</sub>(I<sub>LR</sub>)) - 𝜙(I<sub>HR</sub>)||<sub>2</sub><sup>2</sup>_,

where 𝜙  refers to the 19-layer VGG network. With this loss term, I<sub>SR</sub>  and I<sub>HR</sub> are encouraged to have similar feature representations rather than to exactly match with each other in a pixel wise manner.

# vgg code here

## Adversarial Losses:
In [SRGAN](https://arxiv.org/abs/1609.04802), the adopted generative model is generative adversarial network (GAN) and it suffers from training instability. WGAN leverages the Wasserstein distance to produce a value function, which has better theoretical properties than the original GAN. However, WGAN requires that the discriminator must lie within the space of 1-Lipschitz through weight clipping, resulting in either vanishing or exploding gradients without careful tuning of the clipping threshold.     
To overcome the flaw of clipping , a new approach is applied called Gradient Pelanty method. It is used to enforce the Lipschitz constraint. This way Wasserstein distance between two distributions to help decide when to stop the training but penalizes the gradient of the discriminator with respect to its input instead of weight clipping. With gradient penalty, the discriminator is encouraged to learn smoother decision boundaries. 

### Generator Loss

_l<sub>GA</sub>=-𝔼[D(G<sub>Θ</sub>(I<sub>LR</sub>)]_

### Discriminator Loss

_l<sub>DA</sub>=𝔼[D(G<sub>Θ</sub>(I<sub>LR</sub>)]-𝔼[D(I<sub>HR</sub>)] + λ𝔼(||▽<sub>hat{I}</sub>D(hat{I})-1||<sub>2</sub>-1)<sup>2</sup>_

![workflow diagram](/images/work_flow.png#center)


Normally, the output of the classifier i.e. discriminator in this case is kept between 0-1 using a sigmoid function in the last layer, where if discriminator prediction 0 for an image then the image is SR likewise if the prediction is 1 then it is an HR image. Here the discriminator is trained using WGAN-GP approach [(described here)](https://sulavtimilsina.github.io/posts/wgan-gp/), hence the output is not bounded between 0-1 instead the discriminator will try to maximize the distance between the prediction of SR image and HR image and generator will try to minimize it. Let's look at the loss of the generator ie. I<sub>GA</sub> and the loss of discriminator  I<sub>DA</sub> .

### UNDERSTANDING DISCRIMINATOR ADVERSARIAL LOSS 
(not considering the gradient penalty term for making it easier to understand)

_l<sub>DA</sub>=𝔼[D(G<sub>Θ</sub>(I<sub>LR</sub>)]-𝔼[D(I<sub>HR</sub>)]_

(Note: _l<sub>DA</sub>= 𝔼[D(I<sub>HR</sub>)]-𝔼[D(G<sub>Θ</sub>(I<sub>LR</sub>)]_ if the loss of the discriminator is in this form than the discriminator will try to maximize this equation and the generator will try to minimize the I<sub>GA</sub>).

Considering D(G<sub>Θ</sub>(I<sub>LR</sub>))= 5  and D(I<sub>HR</sub>) = 5 initially when the discriminator doesn’t have the ability to differentiate between them.

Therefore the loss at the very beginning:
_l<sub>DA</sub>=5-5= 0,_ 

The discriminator wants to minimize the loss l<sub>DA</sub>, hence increasing the distance between
D(G<sub>Θ</sub>(I<sub>LR</sub>))and D(I<sub>HR</sub>) . Suppose after the update of the gradient of the discriminator for the few step, the value of prediction becomes D(G<sub>Θ</sub>(I<sub>LR</sub>))=-2   and D(I<sub>HR</sub>) = 2 ,therefore discriminator is learning to know the difference between the LR image and the HR image, hence making the loss(l<sub>DA</sub>)= -4, Here the loss is minimized and the distance between the two predictions is maximized.


![discriminator adverserial loss](/images/understanding_disc_adv_loss.png#center)


# code

### UNDERSTANDING GENERATOR ADVERSARIAL LOSS

Discriminator is trained for a few steps and then the update of the generator happens. Therefore the discriminator is kept a few steps ahead of the generator in terms of its learning. Let's consider the discriminator has been trained for the few steps and it predicted outputs are:

_D(G<sub>Θ</sub>(I<sub>LR</sub>)) = -2_
_D(I<sub>HR</sub>) = 2_ 

The loss of the generator is:

_l<sub>GA</sub> = -𝔼[D(G<sub>Θ</sub>(I<sub>LR</sub>)]_

Therefore,
_l<sub>GA</sub>=  -(-2) = 2_

Generator wants to minimize l<sub>GA</sub> , which can only we achieved by increasing the value of D(G<sub>Θ</sub>(I<sub>LR</sub>)) hence ultimately reducing the distance between D(G<sub>Θ</sub>(I<sub>LR</sub>)) and D(I<sub>HR</sub>) ,hence making the SR image and HR image identical as:

_l<sub>GA</sub>= -(large positive value) ≈ global minima_

![generator adverserial loss](/images/understanding_gen_adv_loss.png#center)


# code

### Result and Conclusion:
We chose Kaggle's kernel with Tesla P100 GPU to train the model. Since it has 20 million training parameters, training it for 500 epochs is a tedious job. Until now we have trained it only up to 100 epochs.
Following is the sample output of the 100th epoch.
The rightmost image is Low-Resolution Patch, the Middle one is the High-Resolution Patch and the Left most one is the Generated High-Resolution Image.

![outputs](/images/output.png#center)

