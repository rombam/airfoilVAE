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

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
        weight = kwargs['weight']
        
        loss = recon_loss + weight*kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recon_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        coordinate space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples