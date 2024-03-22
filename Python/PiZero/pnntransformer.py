import numpy as np
import torch
from torch import nn

import PiZero


class PiNet(nn.Module):

    def __init__(self, C: int, H: int,
                 n_transformer_layers: int, num_moves: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 batch_first: bool = True,
                 norm_first: bool = True):
        """
        Initializes the PiZero network with C channels of height H and width W.
        :param S:
        :param E:
        :param d_model:
        :param num_moves: number of moves over which to output a policy
        """

        super(PiNet, self).__init__()

        self.value_embedding = nn.Embedding(C, d_model)
        self.position_embedding = nn.Embedding(H, d_model)
        self.positional_tensor = torch.from_numpy(
            np.arange(H, dtype=np.int64)[np.newaxis, :]).to(PiZero.DEVICE)

        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.transformer_net = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                activation="gelu",
                batch_first=batch_first, norm_first=norm_first),
            num_layers=n_transformer_layers
        )

        # self.mlp_layer = nn.Sequential(
        #     # nn.LayerNorm(d_model),
        #     nn.Linear(d_model, d_model * 4),
        #     # nn.GELU(),
        #     nn.Linear(d_model * 4, d_model)
        # )

        in_features = d_model  # TODO: adjust when pooling
        self.policy_head = PolicyHead(in_features, num_moves)
        self.value_head = ValueHead(in_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # embed the values and positions of x
        z = self.value_embedding(x)
        p = self.position_embedding(
            torch.cat(tuple(self.positional_tensor for _ in range(x.size(0))),
                      dim=0))

        # add z and p, and prepend the class token
        x = torch.cat((
            torch.cat(tuple(self.class_token for _ in range(x.size(0))), dim=0),
            z + p),
            dim=1
        )

        # pas through transformer
        x = self.transformer_net(x)
        # x = x[:, 0, :]  # only take output corresponding to class embedding
        x = torch.mean(x, dim=1)
        # x = self.mlp_layer(x)
        return self.policy_head(x), self.value_head(x)


class PolicyHead(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        """
        :param in_features
        :param out_features
        """

        super(PolicyHead, self).__init__()

        # hidden_features = (in_features + out_features) * 2
        # self.linear_net = nn.Sequential(
        #     nn.LayerNorm(in_features),
        #     nn.Linear(in_features, hidden_features),
        #     nn.GELU(),
        #     nn.Linear(hidden_features, out_features)
        # )

        self.linear_net = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_net(x)


class ValueHead(nn.Module):

    def __init__(self, in_features: int):
        """
        Initializes the value head with C channels of height H and width W.
        Outputs a single scalar value between 0 and 1.
        :param in_features: int
        :param hidden_features: int
        """

        super(ValueHead, self).__init__()

        # hidden_features = (in_features + 1) * 2
        # self.linear_net = nn.Sequential(
        #     nn.Linear(in_features, hidden_features),
        #     nn.GELU(),
        #     nn.Linear(hidden_features, 1),
        #     nn.GELU()
        # )

        self.linear_net = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_net(x)
