import torch.nn as nn
import torch
import torch.nn.functional as F
class MFB(nn.Module):
    def __init__(self, img_feat_size, ques_feat_size):
        super(MFB, self).__init__()
        self.MFB_K=5
        self.MFB_O=512
        self.DROPOUT_R=0.1
        self.proj_i = nn.Linear(img_feat_size, self.MFB_K * self.MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, self.MFB_K * self.MFB_O)
        self.dropout = nn.Dropout(self.DROPOUT_R)
        self.pool = nn.AvgPool1d(self.MFB_K, stride=self.MFB_K)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat                  # (N, C, K*O)
        exp_out = self.dropout(exp_out)   # (N, C, K*O)
        z = self.pool(exp_out) * self.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.MFB_O)      # (N, C, O)
        return z
    

if __name__ == '__main__':
    x1 = torch.rand((48,512))
    x2 = torch.rand((48,512))
    mfb = MFB(512,512)
    output = mfb(x1,x2)
    print(output.shape)
    # print(exp_out.shape)