#----------------------------#
#   ECA module的PyTorch实现
#----------------------------#

import torch
from torch import nn
from torchsummary import summary


class ECA_Layer(nn.Module):
    """
    Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1, 512, 20, 20 -> 1, 512, 1, 1
        y = self.avg_pool(x)

        # 1, 512, 1, 1 -> 1, 512, 1 -> 1, 1, 512 -> 1, 1, 512 -> 1, 512, 1 -> 1, 512, 1, 1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 1, 512, 1, 1 -> 1, 512, 1, 1
        y = self.sigmoid(y)

        # 看一下权重
        # print(y)

        return x * y.expand_as(x)


model = ECA_Layer(512)
# print(model)
summary(model, input_size=[(1, 1, 2048)], batch_size=1, device="cpu")

inputs = torch.rand([1, 512, 20, 20])
outputs = model(inputs)
print(outputs.shape)
