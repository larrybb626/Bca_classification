import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import warnings
# from networks.transformer import TransformerModel
from networks.SpatialAttention import CBAM
from networks.SELayer import SELayer, SELayer_dual
# from SpatialAttention import CBAM
# from SELayer import SELayer,SELayer_dual
# from networks.cnsn import CNSN, SelfNorm, CrossNorm
# from networks.MFB import MFB
from networks.LBP import LBP
warnings.filterwarnings('ignore')
# from MobileViT import mobilevit_xxs
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


class FilterResponseNormNd(nn.Module):

    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False):
        """
        Input Variables:
        ----------------
            ndim: An integer indicating the number of dimensions of the expected input tensor.
            num_features: An integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        assert ndim in [3, 4, 5], \
            'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim())) # (2, 3)
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        #self.gn1 = nn.GroupNorm(32,planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        #self.gn2 = nn.GroupNorm(32,planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(channel=planes*self.expansion)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.gn2(out)
        out = self.cbam(out)
        # print("Used CBAM")
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        #self.gn1 = nn.GroupNorm(32,planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        #self.gn2 = nn.GroupNorm(32,planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
       #self.gn3 = nn.GroupNorm(32,planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(channel=planes*self.expansion)
    def forward(self, x):
        residual = x
        # print(x.shape)

        out = self.conv1(x)
        # print(11)
        # print(x.shape)
        out = self.bn1(out)
        #out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print(22)
        # print(x.shape)
        out = self.bn2(out)
        #out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # print(33)
        # print(x.shape)
        out = self.bn3(out)
        # print('change')
        # print(out.shape)
        #out = self.gn3(out)
        out = self.cbam(out)
        # print("Used CBAM")
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        # print(out.shape)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.feature = None
        self.num_classes = num_classes
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        #self.gn1 = nn.GroupNorm(32,64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        # self.avgpool = nn.AvgPool3d(
        #     (4, 2, 2), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                # downsample = nn.Sequential(
                #     nn.Conv3d(
                #         self.inplanes,
                #         planes * block.expansion,
                #         kernel_size=1,
                #         stride=stride,
                #         bias=False), nn.GroupNorm(32,planes * block.expansion))
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        #x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        self.feature = x
        x = self.fc(x)
        if self.num_classes == 1:
            x = F.sigmoid(x)
        return x


# def initialize_weights(self):
#         # print(self.modules())
#
#     for m in self.modules():
#         if isinstance(m, nn.Linear):
#                 # print(m.weight.data.type())
#                 # input()
#                 # m.weight.data.fill_(1.0)
#             nn.init.kaiming_normal_(m.weight,a=0, mode='fan_in', nonlinearity='relu')
#             print(m.weight)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # model.apply(weights_init)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # model.apply(weights_init)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    # model.apply(weights_init)
    return model


# todo 将两个模型并联起来 
class CombinedModel(nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.model1 = nn.Sequential(*list(model1.children())[:-1])
        self.model2 = nn.Sequential(*list(model2.children())[:-1])
        # self.encoder = TransformerModel()
        self.SElayer_dual = SELayer_dual(in_channel=2048)
        # self.mySelfNorm = SelfNorm(chan_num=4096)
        # self.myCrossNorm = CrossNorm(crop=0.47,beta=0.47)
        # self.cnsn = CNSN(self.myCrossNorm,self.mySelfNorm)
        self.Dropout_1 = torch.nn.Dropout(p=0.47, inplace=False)
        self.Dropout_2 = torch.nn.Dropout(p=0.2, inplace=False)
        "给x1的全连接层"
        self.fc_x1_1 = nn.Linear(2048, 1024)
        self.fc_x1_2 = nn.Linear(1024, 1)
        "给x2的全连接层"
        self.fc_x2_1 = nn.Linear(2048, 1024)
        self.fc_x2_2 = nn.Linear(1024, 1)
        "给x的全连接层"
        self.fc_x_1 = nn.Linear(2048, 1024)
        self.fc_x_2 = nn.Linear(1024, 1)
        self.fc_choose = nn.Linear(3, 1)

    def forward(self, x1, x2):
        x1 = LBP(x1)
        x2 = LBP(x2)
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x1 = torch.squeeze(x1, dim=-1)
        x2 = torch.squeeze(x2, dim=-1)
        
        x = self.SElayer_dual(x1, x2)
        """对每个分支都计算出预测分数"""
        x1 = x1.view(x1.size(0), -1) 
        x2 = x2.view(x2.size(0), -1) 
        # x  = self.MFB(x1,x2)
        x = x.view(x.size(0), -1) 

        x1 = self.fc_x1_1(x1) 
        x1 = self.fc_x1_2(x1) 

        x2 = self.fc_x2_1(x2) 
        x2 = self.fc_x2_2(x2)
  
        x = self.fc_x_1(x)   
        x = self.Dropout_1(x)
        x = self.fc_x_2(x)
        # x = torch.squeeze(x,dim=-1)
        # x = F.relu(x)

        output = torch.cat((x1, x2, x), dim=1)
        output = self.Dropout_2(output)
        output = self.fc_choose(output)
        output= F.sigmoid(output)
        return output


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else ('cpu')
    input1 = torch.ones([64, 1, 8,128, 128]).to(device)
#     # input1 = torch.squeeze(input1,0)
    input2 = torch.ones([64, 1, 8, 128, 128]).to(device)
#     # input2 = torch.squeeze(input2, 0)
    model = resnet50(num_classes=1,sample_size=128,sample_duration=8)
#     # # # output = model(input)
#     # # # input2 = torch.ones([1, 1, 8, 128, 128])
    model2 = resnet50(num_classes=1, sample_size=128, sample_duration=8)
#     # # output2 = model(input)/
#     # # # print(output.shape)
#     # model1 = mobilevit_xxs()
#     # model2 = mobilevit_xxs()
    combined_model = CombinedModel(model, model2).to(device)
#     # # # # 添加新的全连接层
#     # # # # combined_model.fc = nn.Linear(4096, 2)
    output = combined_model(input1, input2)
    print(output.shape)
#     # # #
#     # # #
#     # #
#     # # print(output.shape)
#     # # _ = [print(i.shape) for i in output]
#     # # print(model(input).feature.shape)