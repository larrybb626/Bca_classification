import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")



class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        nn.TransformerEncoderLayer.__init__().self_attn
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(4096, 1)
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.squeeze(x,dim=1)
        x = torch.transpose(x, dim0=1, dim1=0)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x

if __name__=="__main__":
    model = TransformerModel()
    input_tensor = torch.randn(4096, 1, 24)  # 生成一个随机的输入张量
    output_prob = model(input_tensor)

    print(output_prob.shape)  # 输出概率值，item()方法将输出张量转换为标量
