# AirfoilVAE
 Pytorch implementation of an airfoil generator via a __Variational Autoencoder__.  
 Developed in order to use as a sampler/generator for airfoil shape aeroacoustic optimization problems.  

![sample2](https://user-images.githubusercontent.com/86745238/171267514-efe51a25-2a45-49ab-9243-e2794f257410.png)

 ## Contents
 The root of the repository contains three Jupyter notebooks.  

 - __AirfoilVAE.ipynb__: for general purpose exploration of network architectures and parameters, sampling and plotting airfoils.
 - __AirfoilVAE_hyperOpt.ipynb__: used for optimizing the network's architecture using Bayesian optimization (TPE + HyperBand) through the Optuna package.
 - __AirfoilVAE_opt.ipynb__: used to train the final model with the parameters from the hyperparameter optimization.

Data can be found in _./data/_ and the final script that allows sampling of aifoils through external modification of the latent variables is in _./model/vae_generator.py_.  
Folder _./archive/_ contains test network architectures, previously trained models and other files.  

## References
This work draws heavily from:  

- *Pytorch VAE* - https://github.com/AntixK/PyTorch-VAE  
- Kingma, D., Welling, M. *Auto-Encoding Variational Bayes* - https://arxiv.org/abs/1312.6114
- Asperti, A., Trentin, M. *Balancing reconstruction error and Kullback-Leibler divergence in Variational Autoencoders* - https://arxiv.org/abs/2002.07514

