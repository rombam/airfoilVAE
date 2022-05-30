# AirfoilVAE
 Pytorch implementation of an airfoil generator via a __Variational Autoencoder__.  
 Developed in order to use as a sampler/generator for airfoil shape aeroacoustic optimization problems.  

 ## Contents
 The root of the repository contains three Jupyter notebooks.  
 
 - __AirfoilVAE.ipynb__: for general purpose exploration of network architectures and parameters, sampling and plotting airfoils.
 - __AirfoilVAE_hyperOpt.ipynb__: used for optimizing the network's architecture using Bayesian optimization (TPE + HyperBand) through the Optuna package.
 - __AirfoilVAE_opt.ipynb__: used to train the final model with the parameters from the hyperparameter optimization.

Data can be found in __./data/__ and the final script that allows sampling of aifoils through external modification of the latent variables is in __./model/vae_generator.py__.  
Folder __./archive/__ contains test network architectures, previously trained models and other files.  

## Acknowledgements
This work draws heavily from:  

- *Pytorch VAE* - https://github.com/AntixK/PyTorch-VAE  
- Kingma, D., Welling, M. *Auto-Encoding Variational Bayes* - https://arxiv.org/abs/1312.6114
- Asperti, A., Trentin, M. *Balancing reconstruction error and Kullback-Leibler divergence in Variational Autoencoders* - https://arxiv.org/abs/2002.07514

