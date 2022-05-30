import torch
from torch import nn
from torch.nn import functional as F


class TwoStageVAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims = None,
                 hidden_dims2= None,
                 act_function = nn.ELU(),
                 **kwargs):
        super(TwoStageVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        if hidden_dims2 is None:
            hidden_dims2 = [1024, 1024]
            
        self.name = f'TSVAE_MLP{hidden_dims[0]}_{in_channels}_{latent_dim}'
        
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

        #---------------------- Second VAE ---------------------------#
        encoder2 = []
        in_channels = self.latent_dim
        for h_dim in hidden_dims2:
            encoder2.append(nn.Sequential(
                                nn.Linear(in_channels, h_dim),
                                nn.BatchNorm1d(h_dim),
                                nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder2 = nn.Sequential(*encoder2)
        self.fc_mu2 = nn.Linear(hidden_dims2[-1], self.latent_dim)
        self.fc_var2 = nn.Linear(hidden_dims2[-1], self.latent_dim)

        decoder2 = []
        hidden_dims2.reverse()

        in_channels = self.latent_dim
        for h_dim in hidden_dims2:
            decoder2.append(nn.Sequential(
                                nn.Linear(in_channels, h_dim),
                                nn.BatchNorm1d(h_dim),
                                nn.LeakyReLU()))
            in_channels = h_dim
        self.decoder2 = nn.Sequential(*decoder2)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

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

        kld_weight = kwargs['weight'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'recons_loss': recons_loss.detach(), 'kl_loss': kld_loss.detach()}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]