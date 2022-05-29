import numpy as np
import time
import json
import torch
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
    airfoil = denormalize(decoded_airfoil)
    print('Decoding latent variables...')
    return airfoil

def denormalize(data):
    """
    Denormalizes numpy array data using a pre-fitted scaler saved in scaler_dict.json as a dictionary.
    Inputs:
        - data: data to be normalized. [np.array]
    Outputs:
        - normalized data. [np.array]
    """
    with open('./params/scaler_dict.json', 'r') as fp:
        scaler_bounds = json.load(fp)
    idx = 0
    data_norm = data
    
    for key in scaler_bounds.keys():
        data_norm[idx] = (data[idx] + scaler_bounds[key]['min']) * (scaler_bounds[key]['max'] - scaler_bounds[key]['min'])
        idx += 1
    return data_norm

def read_latent(filename):
    """
    Reads an airfoil's latent variables from a .dat file.
    Inputs:
    - filename: string of the filename to read the airfoil from
    Outputs:    
    - latent: numpy array with the latent variables
    """
    with open(filename, 'r') as datfile:
        print(f'Reading airfoil from: {filename}')
        latent = np.loadtxt(datfile, unpack = True)
    return latent

def save_airfoil(airfoil, filename, n_points = 198):
    """
    Saves an airfoil's x and y coordinates to a .dat file. Uses cosine spacing.
    Inputs:
    - airfoil: numpy array of Y airfoil coordinates
    - filename: string of the filename to save the airfoil to
    """

    # X: cosine spacing
    points_per_surf = int(n_points/2)
    x = list(reversed([0.5*(1-np.cos(ang)) for ang in np.linspace(0,np.pi,points_per_surf+2)]))
    aux_x = list([0.5*(1-np.cos(ang)) for ang in np.linspace(0,np.pi,points_per_surf+2)[1:points_per_surf+1]])
    [x.append(i) for i in aux_x]
    x.append(1.0)
    
    # Y
    y = []
    origin = (airfoil[0] + airfoil[points_per_surf])/2
    y.append(0.0)
    [y.append(j) for j in airfoil[0:points_per_surf].tolist()]
    y.append(origin)
    aux_y = list(airfoil[points_per_surf:n_points].tolist())
    [y.append(k) for k in aux_y]
    y.append(0.0)
    
    with open(filename, 'w', newline='') as datfile:
        for i in range(len(x)):
            print(f'{x[i]:.8f} {y[i]:.8f}', file=datfile)
    
    print('Airfoil saved successfully!')
    
if __name__ == '__main__':
    print('--- VAE Airfoil Generator ---\n')
    
    # Load input parameters and model
    start_time = time.time()
    try:
        with open('./params/model_parameters.json', 'r') as f:
            parameters = json.load(f)
    except:
        print('There was an error loading model_parameters.json. Check your inputs.')
    print(parameters)
    model = load_model(parameters)

    # Decode and transform the input latent variables
    latent_airfoil = read_latent('input_latent.dat')
    latent_tensor = torch.Tensor(latent_airfoil)
    decoded_coords = decode_latent(latent_tensor, model, device = parameters['device'])
    airfoil_coords = denormalize(decoded_coords)
    save_airfoil(airfoil_coords, 'output_airfoil.dat')
    print(f'Elapsed time: {np.round(time.time() - start_time, 4)} s')