# Weakly Supervised Disentanglement by Pairwise Similarities

This repository contains the PyTorch source code for the paper [Weakly Supervised Disentanglement by Pairwise Similarities](https://arxiv.org/abs/1906.01044) by Junxiang Chen and Kayhan Batmanghelich. 


## Model

In this paper, we propose a setting where the user introduces weaksupervision by providing similarities between instances (denoted by $y_{ij}$) based on a factor  to  be  disentangled. The  similarity  is  providedas  either  a  binary  (yes/no)  or  real-valued  label  describingwhether a pair of instances are similar or not. We propose a new method for weakly supervised disentanglement of latent variables within the framework of Variational Autoencoder.

<p align="center">
  <img width="95%" src="https://github.com/batmanlab/VAE_pairwise/blob/master/figure/model.jpg">
</p>


## Environment 
To prepare for the environment for running our code, run 

```conda env create -f VAE_pairwise.yml```

# Experiments
We include an example of training script in test.ipynb. 
## MNIST with binary pairwise labels
We train the model with binary pairwise labels for the MNIST dataset. The embedding and generated results are shown below:

<div class="row">
  <div class="column">
    <img width="45%" src="https://github.com/batmanlab/VAE_pairwise/blob/master/figure/binary_embedding.png">  </div>
  <div class="column">
    <img width="45%" src="https://github.com/batmanlab/VAE_pairwise/blob/master/figure/binary_generated.png">
  </div>


<img width="45%" src="https://github.com/batmanlab/VAE_pairwise/blob/master/figure/binary_embedding.png">

<img width="45%" src="https://github.com/batmanlab/VAE_pairwise/blob/master/figure/binary_generated.png">


## MNIST with real-valued pairwise labels
We also train the model with real-valued pairwise labels for the MNIST dataset. This setting is for illustration purposes only, but might not be useful in solving real-world problems. The embedding and generated results are shown below:
<img width="45%" src="https://github.com/batmanlab/VAE_pairwise/blob/master/figure/real_embedding.png">

<img width="45%" src="https://github.com/batmanlab/VAE_pairwise/blob/master/figure/real_generated.png">


# Citation

```
@article{chen2019weakly,
  title={Weakly Supervised Disentanglement by Pairwise Similarities},
  author={Chen, Junxiang and Batmanghelich, Kayhan},
  journal={arXiv preprint arXiv:1906.01044},
  year={2019}
}
```

# Acknowledgments

This work was partially supported by NIH Award Number 1R01HL141813-01, NSF 1839332 Tripod+X, and SAP SE. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan X Pascal GPU used for this research. We were also grateful for the computational resources provided by Pittsburgh SuperComputing grant number TG-ASC170024.
