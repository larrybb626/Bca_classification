from networks.unet3d import UNet3D
import torch.nn.functional as F
from torch import nn
import torch

def MY_Unet():
    class TargetNet(nn.Module):
        def __init__(self, base_model,n_class=1):
            super(TargetNet, self).__init__()

            self.base_model = base_model
            self.dense_1 = nn.Linear(512, 1024, bias=True)
            self.dense_2 = nn.Linear(1024, n_class, bias=True)

        def forward(self, x):
            self.base_model(x)
            self.base_out = self.base_model.out512
            # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
            # where N = batch_size, C = channels, H = height, and W = Width
            self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
            self.linear_out = self.dense_1(self.out_glb_avg_pool)
            final_out = self.dense_2( F.relu(self.linear_out))
            final_out = F.sigmoid(final_out)
            return final_out

    base_model = UNet3D()
    weight_dir = './Genesis_Chest_CT.pt'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)
    model = TargetNet(base_model)

    return model


def MY_Unet_multi():
    class TargetNet(nn.Module):
        def __init__(self, base_model, n_class=1):
            super(TargetNet, self).__init__()

            self.base_model = base_model
            self.dense_1 = nn.Linear(512, 1024, bias=True)
            self.dense_2 = nn.Linear(1024, n_class, bias=True)
            self.dense_3 = nn.Linear(512, 1024, bias=True)
            self.dense_4 = nn.Linear(1024, n_class, bias=True)
            self.dense_5 = nn.Linear(512, 1024, bias=True)
            self.dense_6 = nn.Linear(1024, n_class, bias=True)
            self.dense_7 = nn.Linear(512, 1024, bias=True)
            self.dense_8 = nn.Linear(1024, n_class, bias=True)



        def forward(self, x):
            self.base_model(x)
            self.base_out = self.base_model.out512
            # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
            # where N = batch_size, C = channels, H = height, and W = Width
            self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
            self.linear_out = self.dense_1(self.out_glb_avg_pool)
            final_out = self.dense_2( F.relu(self.linear_out))
            final_out = F.sigmoid(final_out)

            self.linear_out2 = self.dense_3(self.out_glb_avg_pool)
            final_out2 = self.dense_4(F.relu(self.linear_out2))
            final_out2 = F.sigmoid(final_out2)

            self.linear_out3 = self.dense_5(self.out_glb_avg_pool)
            final_out3 = self.dense_6(F.relu(self.linear_out3))
            final_out3 = F.sigmoid(final_out3)

            # self.linear_out4 = self.dense_7(self.out_glb_avg_pool)
            # final_out4 = self.dense_8(F.relu(self.linear_out4))
            # final_out4 = F.sigmoid(final_out4)




            return final_out, final_out2, final_out3 #, final_out4

    def MY_Unet_tasks():
        class TargetNet(nn.Module):
            def __init__(self, base_model, n_class=1):
                super(TargetNet, self).__init__()

                self.base_model = base_model
                self.dense_1 = nn.Linear(512, 1024, bias=True)
                self.dense_2 = nn.Linear(1024, n_class, bias=True)


            def forward(self, x):
                self.base_model(x)
                self.base_out = self.base_model.out512
                # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
                # where N = batch_size, C = channels, H = height, and W = Width
                self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(
                    self.base_out.size()[0], -1)
                self.linear_out = self.dense_1(self.out_glb_avg_pool)
                final_out = self.dense_2(F.relu(self.linear_out))
                final_out = F.sigmoid(final_out)


                return final_out, self.base_model.out

    base_model = UNet3D()
    weight_dir = './Genesis_Chest_CT.pt'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)
    # model = TargetNet(base_model)
    model = MY_Unet_tasks()

    return model

