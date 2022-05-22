import torch
from torch import nn
from torch.nn import functional as F


class AirfoilInfoVAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims = None,
                 alpha = -0.5,
                 beta = 5.0,
                 reg_weight = 100,
                 kernel_type = 'imq',
                 latent_var = 2.,
                 **kwargs):
        super(AirfoilInfoVAE, self).__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        assert alpha <= 0, 'alpha must be negative or zero.'

        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Build Encoder
        modules = []
        modules.append(nn.Linear(in_channels, hidden_dims[0]))
        modules.append(nn.ELU())
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.ELU())
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
                    nn.ELU())
            )
        modules.append(nn.Linear(hidden_dims[-1], in_channels))
        self.decoder = nn.Sequential(*modules)

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
        return  [self.decode(z), input, z, mu, log_var]

    def loss_function(self,
                      pred,
                      *args,
                      **kwargs) -> dict:
        recons = pred[0]
        input = pred[1]
        z = pred[2]
        mu = pred[3]
        log_var = pred[4]

        batch_size = input.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        kld_weight = kwargs['weight']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = self.beta * recons_loss + \
               (1. - self.alpha) * kld_weight * kld_loss + \
               (self.alpha + self.reg_weight - 1.)/bias_corr * mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'MMD': mmd_loss, 'KLD':-kld_loss}

    def compute_kernel(self,
                       x1,
                       x2):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result


    def compute_rbf(self,
                    x1,
                    x2,
                    eps = 1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                               x1,
                               x2,
                               eps: float = 1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z):
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
              z__kernel.mean() - \
              2 * priorz_z__kernel.mean()
        return mmd

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
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