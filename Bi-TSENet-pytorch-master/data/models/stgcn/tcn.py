import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, seq_len, num_features]
        Returns: Output tensor of shape [batch_size, seq_len, num_channels[-1]]
        """
        # Convert [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        x = self.network(x)
        # Convert back to [batch, seq_len, features]
        return x.transpose(1, 2)


class BiDirectionalTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(BiDirectionalTCN, self).__init__()
        self.forward_tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.backward_tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)

        # Output layer to combine both directions
        self.fc = nn.Linear(num_channels[-1] * 2, num_channels[-1])

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, seq_len, num_features]
        """
        # Forward direction
        forward_out = self.forward_tcn(x)

        # Backward direction (reverse sequence)
        backward_in = torch.flip(x, [1])
        backward_out = self.backward_tcn(backward_in)
        backward_out = torch.flip(backward_out, [1])

        # Concatenate both directions
        combined = torch.cat([forward_out, backward_out], dim=2)

        # Final output projection
        return self.fc(combined)