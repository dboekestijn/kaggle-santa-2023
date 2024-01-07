import torch
from torch import nn


class PiNet(nn.Module):

    def __init__(self, C: int, H: int, W: int, num_moves: int,
                 hidden_channels: int = 64,
                 num_resblocks: int = 10):
        """
        Initializes the PiZero network with C channels of height H and width W.
        :param C: number of channels per facelet value
        :param H: height of each input channel
        :param W: width of each input channel
        :param num_moves: number of moves over which to output a policy
        :param hidden_channels: number of hidden channels in all
        convolutional and residual layers of the network
        :param num_resblocks: number of residual blocks in the network
        """

        super(PiNet, self).__init__()

        self.resnet = nn.Sequential(
            ConvBlock(C, hidden_channels),
            *[ResBlock(hidden_channels, hidden_channels)
              for _ in range(num_resblocks)],
        )

        self.policy_head = PolicyHead(hidden_channels, H, W, num_moves)
        self.value_head = ValueHead(hidden_channels, H, W)

    def policy_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.resnet(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.resnet(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
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

    def __init__(self, C: int, H: int, W: int, num_moves: int,
                 out_channels: int = 2):
        """
        Initializes the policy head with C channels of height H and width W.
        The number of moves specifies the number of output features.
        :param C: number of input channels
        :param H: height of each input channel
        :param W: width of each input channel
        :param num_moves: number of moves over which to output a policy
        :param out_channels: number of output channels from the (first and
        only) convolutional layer
        """

        super(PolicyHead, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(C, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        linear_input_features = out_channels * H * W
        self.linear_net = nn.Linear(linear_input_features, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        return self.linear_net(x)


class ValueHead(nn.Module):

    def __init__(self, C: int, H: int, W: int,
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
            nn.ReLU(inplace=True),
        )

        linear_input_features = out_channels * H * W
        self.linear_net = nn.Sequential(
            nn.Linear(linear_input_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        return self.linear_net(x)
