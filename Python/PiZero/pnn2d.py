import torch
from torch import nn


class PiNet(nn.Module):

    def __init__(self, C: int, H: int, W: int,
                 num_moves: int, num_distances: int,
                 hidden_channels: int = 64,
                 num_resblocks: int = 10):
        """
        Initializes the PiZero network with C channels of height H and width W.
        :param C: number of channels per input 'image'
        :param H: height of each input channel
        :param W: width of each input channel
        :param num_moves: number of moves over which to output a policy
        :param hidden_channels: number of hidden channels in all
        convolutional and residual layers of the network
        :param num_resblocks: number of residual blocks in the network
        """

        super(PiNet, self).__init__()

        self.shared_embedder = nn.Sequential(
            ConvBlock(C, hidden_channels),
            *[ResBlock(hidden_channels, hidden_channels)
              for _ in range(num_resblocks)],
            PolicyHead(hidden_channels, H, W, num_moves, hidden_channels),
            # nn.LeakyReLU(inplace=True)
        )

        linear_input_features = num_moves
        self.dist_net = nn.Sequential(
            nn.Linear(linear_input_features, linear_input_features // 2),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(linear_input_features // 2, 1),
            nn.LeakyReLU(inplace=True)
        )

        # self.policy_head = PolicyHead(hidden_channels, H, W, num_moves)
        # self.value_head = ValueHead(hidden_channels, H, W, num_distances)
        # self.value_head = PolicyHead(hidden_channels, H, W, num_distances)  # TODO: NB!

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1, e2 = self.shared_embedder(x1), self.shared_embedder(x2)
        # dist_output = torch.sum(torch.abs(e1 - e2), dim=1)

        dist_output = self.dist_net(torch.abs(e1 - e2))

        return dist_output


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=0 if kernel_size == 1 else 1,
                      stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            self.relu,
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x) + x
        return self.relu(x)


class PolicyHead(nn.Module):

    def __init__(self, C: int, H: int, W: int, out_features: int,
                 out_channels: int = 2):
        """
        Initializes the policy head with C channels of height H and width W.
        The number of moves specifies the number of output features.
        :param C: number of input channels
        :param H: height of each input channel
        :param W: width of each input channel
        :param out_features: number of moves over which to output a policy
        :param out_channels: number of output channels from the (first and
        only) convolutional layer
        """

        super(PolicyHead, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(C, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

        linear_input_features = out_channels * H * W
        self.linear_net = nn.Linear(linear_input_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        return self.linear_net(x)


class ValueHead(nn.Module):

    def __init__(self, C: int, H: int, W: int, num_distances: int,
                 out_channels: int = 1,
                 hidden_features: int = 64):
        """
        Initializes the value head with C channels of height H and width W.
        Outputs a single scalar value between 0 and 1.
        :param C: number of input channels
        :param H: height of each input channel
        :param W: width of each input channel
        :param out_channels: number of output channels from the (first and
        only) convolutional layer
        :param hidden_features: number of hidden features in the two linear
        layers
        """

        super(ValueHead, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(C, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

        linear_input_features = out_channels * H * W
        self.linear_net = nn.Sequential(
            nn.Linear(linear_input_features, hidden_features),
            nn.Linear(hidden_features, 1),
            nn.LeakyReLU(inplace=True)
            # nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        return self.linear_net(x)
