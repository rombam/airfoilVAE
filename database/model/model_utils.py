import torch
import numpy as np
from model.AirfoilVAE import AirfoilVAE

def load_model(parameters):
    """
    Loads the pre-trained Pytorch model given an input parameter dictionary.
    Inputs:
    - parameters: dictionary of parameters (in_channels, latent_dim, hidden_dim, device)
    """
    
    # Unroll the input parameter dictionary
    in_channels     = parameters['in_channels']
    latent_dim      = parameters['latent_dim']
    hidden_dims     = parameters['hidden_dims']
    device          = parameters['device']
    epochs          = parameters['epochs']
    kld_weight_coef = parameters['kld_weight_coef']
    batch_size      = parameters['batch_size']
    
    model = AirfoilVAE(in_channels = in_channels,
                latent_dim = latent_dim,
                hidden_dims = hidden_dims).to(device)

    try:
        model_root = "./model/"
        model_name = f'{model.name}_{epochs}ep_k{kld_weight_coef}_b{batch_size}.pth'
        print(f'Loading model from: {model_root + model_name}')
        model_path = model_root + model_name
        print('Model loaded successfully!\n')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e: 
        print('There was an error loading the model.')
        print(e)
    
    return model

def decode_latent(z, model, device):
    """
    Decodes the input latent variables and returns airfoil coordinates in numpy format.
    Inputs:
    - z: numpy array containing latent variables to decode
    - model: Pytorch model to use for decoding
    Outputs:
    - decoded_airfoil: numpy array of decoded airfoil coordinates
    """
    
    # Decode the latent variables
    z = torch.Tensor(z).to(device)
    decoded_airfoil = model.decode(z).detach().cpu().numpy()
    print('Decoding latent variables...')
    
    return decoded_airfoil

def forward_pass(x, model, device):
    """
    Does a forward pass of the input coordinates.
    Inputs:
    - x: numpy array containing airfoil coordinates
    - model: Pytorch model to use for the forward pass
    - device: Pytorch device to use for the forward pass
    Outputs:
    - recons: list of Pytorch tensors containing the reconstructed airfoil, the original one, mu and log_var.
    """
    # Decode the latent variables
    airfoil = []
    airfoil.extend(x[1:100])
    airfoil.extend(x[101:200])
    
    x_input = torch.Tensor(np.array(airfoil).reshape(1, -1)).to(device)
    recons = model.forward(x_input)
    print('Doing a forward pass...')
    
    return recons