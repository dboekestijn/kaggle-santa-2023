import torch
from torch import nn


class PiNet(nn.Module):

    def __init__(self, in_features, num_moves: int,
                 *hidden_layers: int):
        """
        Initializes the PiZero network with C channels of height H and width W.
        :param in_features: number of input features
        :param num_moves: number of moves over which to output a policy
        :param hidden_layers: variable-length list of the number of output
        features in each hidden layer. if none provided, uses in_features as
        the number of output features for the first linear layer
        """

        super(PiNet, self).__init__()

        self.embedder = nn.Sequential(
            nn.Linear(in_features, in_features ** 2),
            nn.Linear(in_features ** 2, in_features),
            nn.Sigmoid()
        )

        if len(hidden_layers) == 0:
            self.out_features = in_features
            self.linear_net = nn.Sequential(
                nn.Linear(in_features, self.out_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.out_features = hidden_layers[-1]
            self.linear_net = nn.Sequential(
                nn.Linear(in_features, hidden_layers[0]),
                nn.ReLU(inplace=True),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_layers[i], h),
                        nn.ReLU(inplace=True)
                    ) for i, h in enumerate(hidden_layers[1:])
                ]
            )

        self.policy_head = nn.Linear(self.out_features, num_moves)
        self.value_head = nn.Sequential(
            nn.Linear(self.out_features, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> \
            tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embedder(x)
        x = self.linear_net(x)
        return self.policy_head(x), self.value_head(x)


class ResBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()

        self.res1 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
        self.res2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(x) + x)


# class PolicyHead(nn.Module):
#
#     def __init__(self, C: int, H: int, W: int, out_features: int,
#                  out_channels: int = 2):
#         """
#         Initializes the policy head with C channels of height H and width W.
#         The number of moves specifies the number of output features.
#         :param C: number of input channels
#         :param H: height of each input channel
#         :param W: width of each input channel
#         :param out_features: number of moves over which to output a policy
#         :param out_channels: number of output channels from the (first and
#         only) convolutional layer
#         """
#
#         super(PolicyHead, self).__init__()
#
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(C, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             # nn.ReLU(inplace=True),
#         )
#
#         linear_input_features = out_channels * H * W
#         self.linear_net = nn.Linear(linear_input_features, out_features)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv_net(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         return self.linear_net(x)
#
#
# class ValueHead(nn.Module):
#
#     def __init__(self, C: int, H: int, W: int, num_distances: int,
#                  out_channels: int = 1,
#                  hidden_features: int = 64):
#         """
#         Initializes the value head with C channels of height H and width W.
#         Outputs a single scalar value between 0 and 1.
#         :param C: number of input channels
#         :param H: height of each input channel
#         :param W: width of each input channel
#         :param out_channels: number of output channels from the (first and
#         only) convolutional layer
#         :param hidden_features: number of hidden features in the two linear
#         layers
#         """
#
#         super(ValueHead, self).__init__()
#
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(C, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             # nn.ReLU(inplace=True),
#         )
#
#         linear_input_features = out_channels * H * W
#         self.linear_net = nn.Sequential(
#             nn.Linear(linear_input_features, hidden_features),
#             nn.Linear(hidden_features, 1),
#             nn.LeakyReLU(inplace=True)
#             # nn.Sigmoid()
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv_net(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         return self.linear_net(x)