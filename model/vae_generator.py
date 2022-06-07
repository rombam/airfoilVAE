import numpy as np
import time
import json
import torch
from torch import nn
from torch.nn import functional as F
class AirfoilVAE(nn.Module):
    """
    Variational autoencoder designed to reduce dimensionality of airfoil data by encoding their
    coordinates into a latent space, being able to reconstruct them from that latent space.
    Adapted from the base VAE in https://github.com/AntixK/PyTorch-VAE.
    Inputs:
    - in_channels (int): Number of coordinate points in the airfoil data.
    - latent_dim (int): Dimensionality of the latent space.
    - hidden_dims [int]: List of hidden dimensions for the encoder and decoder. Assumed symmetrical.
    - act_function (Callable): Activation function to use for the encoder and decoder.
    Outputs:
    - self (object): Instance of the AirfoilVAE class.
    """
    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims = None,
                 act_function = nn.ELU(),
                 **kwargs):
        super(AirfoilVAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        self.name = f'VAE_MLP{hidden_dims[0]}_{in_channels}_{latent_dim}'
          
        # Build Encoder
        modules = []
        modules.append(nn.Linear(in_channels, hidden_dims[0]))
        modules.append(act_function)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    act_function) 
            )        
        self.encoder = nn.Sequential(*modules)
        
        # Latent variable distributions
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    act_function)
            )
        modules.append(nn.Linear(hidden_dims[-1], in_channels))
        self.decoder = nn.Sequential(*modules)
        
    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x D_in]
        :return: (Tensor) List of latent codes
        """
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        return [mu, log_var]
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D_latent]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D_latent]
        :return: (Tensor) [B x D_latent]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def decode(self, z):
        """
        Maps the given latent codes
        onto the coordinate space.
        :param z: (Tensor) [B x D_latent]
        :return: (Tensor) [B x D_out]
        """
        decoded = self.decoder_input(z)
        decoded = self.decoder(decoded)
        return decoded

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]
    
    def loss_function(self, 
                      pred,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = pred[0]
        input = pred[1]
        mu = pred[2]
        log_var = pred[3]
        
        recon_loss =F.mse_loss(recons, input)

        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
        weight = kwargs['weight']
        
        loss = recon_loss + weight*kl_loss
        return {'loss': loss, 'recon_loss':recon_loss.detach(), 'kl_loss':-kl_loss.detach()}

    def sample(self,
               num_samples,
               current_device,
               std_coef = 1.0,
               **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        mean = torch.zeros(num_samples, self.latent_dim)
        std = torch.ones(num_samples, self.latent_dim)*std_coef
        z = torch.normal(mean, std)

        z = z.to(current_device)
        samples = self.decode(z)
        return samples

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

def encode_latent(x, model, device):
    """
    Decodes the input latent variables and returns airfoil coordinates in numpy format.
    Inputs:
    - z: numpy array containing latent variables to decode
    - model: Pytorch model to use for decoding
    Outputs:
    - decoded_airfoil: numpy array of decoded airfoil coordinates
    """
    # Decode the latent variables
    airfoil = []
    airfoil.extend(x[1:100])
    airfoil.extend(x[101:200])
    
    x_input = torch.Tensor(np.array(airfoil).reshape(1, -1)).to(device)
    latent_mu = model.encode(x_input[0])[0].detach().cpu().numpy()
    latent_std = model.encode(x_input[0])[1].detach().cpu().numpy()
    print('Encoding airfoil...')
    return [latent_mu, latent_std]

def normalize(data):
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
        data_norm[idx] = (data[idx] - scaler_bounds[key]['min']) / (scaler_bounds[key]['max'] - scaler_bounds[key]['min'])
        idx += 1
    return data_norm

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
        data_norm[idx] = data[idx] * (scaler_bounds[key]['max'] - scaler_bounds[key]['min']) + scaler_bounds[key]['min']
        idx += 1
    return data_norm

def read_latent(filename):
    """
    Reads an airfoil's latent variables from a .dat file.
    Inputs:
    - filename: string of the filename to read the airfoil latent variables from
    Outputs:    
    - latent: numpy array with the latent variables
    """
    with open(filename, 'r') as datfile:
        print(f'Reading airfoil from: {filename}')
        latent = np.loadtxt(datfile, unpack = True)
    return latent

def read_airfoil(filename):
    """
    Read an airfoil coordinates from a .dat file.
    Inputs:
    - filename: string of the filename to read the airfoil from
    Outputs:    
    - airfoil: numpy array with the airfoil coordinates [np.array]
    """
    with open(filename, 'r') as datfile:
        print(f'Reading airfoil coordinates from: {filename}')
        airfoil = np.loadtxt(datfile, unpack = True)

    return airfoil[1]

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
    airfoil = denormalize(airfoil)
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
            print(f'{x[i]:.8E} {y[i]:.8E}', file=datfile)
    
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

    # --- Encode airfoil and save latent variables and reconstructed airfoil ---
    # airfoil_shape = read_airfoil('shape_optim.dat')
    # airfoil_shape = normalize(airfoil_shape)
    # airfoil_latent = encode_latent(airfoil_shape, model, parameters['device'])
    # print(airfoil_latent)
    # # Save latent variable
    # with open('optim_latent.dat', 'w', newline='') as datfile:
    #     for i in range(len(airfoil_latent[0])):
    #         print(f'{airfoil_latent[0][i]:.8E}', file=datfile)
    
    # latent_tensor = torch.Tensor(airfoil_latent[0])
    # airfoil_coords = decode_latent(latent_tensor, model, device = parameters['device'])
    # save_airfoil(airfoil_coords, 'output_airfoil_optim.dat')
    # print(f'Elapsed time: {np.round(time.time() - start_time, 4)} s')


    # Decode and transform the input latent variables
    latent_airfoil = read_latent('input_latent.dat')
    latent_tensor = torch.Tensor(latent_airfoil)
    airfoil_coords = decode_latent(latent_tensor, model, device = parameters['device'])
    save_airfoil(airfoil_coords, 'output_airfoil.dat')
    print(f'Elapsed time: {np.round(time.time() - start_time, 4)} s')
    