import torch
from torch import nn
import torch.nn.functional as F

# Define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self,
                 features = 3):
        super(LinearVAE, self).__init__()
        self.features = features
        # encoder
        self.enc1 = nn.Linear(in_features=30, out_features=15)
        self.enc2 = nn.Linear(in_features=15, out_features=self.features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=15)
        self.dec2 = nn.Linear(in_features=15, out_features=30)
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = self.dec2(x)
        return reconstruction, mu, log_var