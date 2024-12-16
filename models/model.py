import torch
from typing import List, Any
from torch import nn, Tensor
from torch.nn import functional as F

from models import BaseVae


class CVae(BaseVae):

    def __init__(self, in_channels: int, num_classes: int, embed_dim: int, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(CVae, self).__init__()

        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.condition = nn.Embedding(num_classes, embed_dim)
        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4 + embed_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4 * 4 + embed_dim, latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim + embed_dim, hidden_dims[-1] * 4 * 4)
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, input_data: Tensor, data_label: Tensor) -> List[Tensor]:
        label_embed = self.condition(data_label)
        result = self.encoder(input_data)
        result = torch.flatten(result, start_dim=1)
        result = torch.cat([result, label_embed], dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, input_data: Tensor, data_label: Tensor) -> Tensor:
        label_embed = self.condition(data_label)
        input_data = torch.cat([input_data, label_embed], dim=1)
        result = self.decoder_input(input_data)
        result = result.view(-1, 128, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def re_parameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, input_data: Tensor, data_label: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input_data, data_label)
        z = self.re_parameterize(mu, log_var)

        return [self.decode(z, data_label), input_data, mu, log_var]

    def loss_function(self, *args: Any, **kwargs) -> dict:
        recons = args[0]
        input_data = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input_data)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
