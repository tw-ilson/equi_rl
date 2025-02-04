import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import torch_utils
from .sac_net import SACGaussianPolicyBase

def pointnet_regularization(tmatA, tmatB):
    batch_size = tmatA.shape[0]
    idA = torch.eye(tmatA.shape[1], requires_grad=True).repeat(batch_size, 1, 1).to(tmatA.device)
    idB = torch.eye(tmatB.shape[1], requires_grad=True).repeat(batch_size, 1, 1).to(tmatB.device)
    diffA = tmatA - torch.bmm(tmatA, tmatA.transpose(1,2))
    diffB = tmatB - torch.bmm(tmatB, tmatB.transpose(1,2))
    pn_reg = 1e-4 * (torch.norm(diffA) + torch.norm(diffB)) / float(batch_size)
    return pn_reg

class TNet(nn.Module):
   def __init__(self, dim=3):
      super().__init__()
      self.dim=dim
      self.conv1 = nn.Conv1d(dim,64,1) # mlp applied point-wise
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,dim*dim)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, x):
      bs, c, n = x.shape

      xb = F.relu(self.bn1(self.conv1(x)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      # input-wise trainable params
      matrix = self.fc3(xb).view(-1,self.dim,self.dim) 
      bias = torch.eye(self.dim, requires_grad=True).repeat(bs,1,1).to(matrix.device)
      return matrix + bias

class TNetBlock(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.dim=dim
        self.tnet = TNet(dim)

    def forward(self, x):
        tmatrix = self.tnet(x)
        out = torch.bmm(x.permute(0,2,1), tmatrix).permute(0,2,1)
        return out, tmatrix


class PointNetBackbone(nn.Module):
    def __init__(self, in_channels=3, global_dim=1024) -> None:
        super().__init__()
        self.global_dim = global_dim
        self.input_transform = TNetBlock(dim=in_channels)
        self.feature_transform = TNetBlock(dim=64)
        # MLP layers ("shared")
        self.conv1 = nn.Conv1d(in_channels,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,global_dim,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(global_dim)

        # output layers
    def forward(self, x):
        b, n, a = x.shape
        x = x.permute(0, 2, 1)
        x, tmatrix3 = self.input_transform(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x, tmatrix64 = self.feature_transform(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=n)
        x = x.flatten(1)
        assert(x.shape == (b, self.global_dim))
        return x, tmatrix3, tmatrix64

class QNetMLP(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim_out)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointQNet(nn.Module):
    def __init__(self, n_p=2, n_theta=1, in_channels=3) -> None:
        super().__init__()
        self.n_inv = 3 * n_theta * n_p
        latent_dim = 1024
        self.backbone = PointNetBackbone(in_channels=in_channels, global_dim=latent_dim)
        self.qnet = QNetMLP(dim_in=latent_dim, dim_out=(self.n_inv*9))

    def forward(self, x):
        B, N, C = x.shape
        x, tmat3, tmat64 = self.backbone(x)
        q = self.qnet(x)
        q = q.reshape(B, self.n_inv, 9).permute(0, 2, 1) # create Q-MAP
        return q, tmat3, tmat64

class SACCriticPointNet(nn.Module):
    
    def __init__(self, in_channels=3, action_dim=5) -> None:
        super().__init__()
        self.backbone = PointNetBackbone(in_channels=in_channels, global_dim=128)
        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, x, act):
        pn_out, tmatA, tmatB = self.backbone(x)
        out_1 = self.critic_fc_1(torch.cat((pn_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((pn_out, act), dim=1))
        return out_1, out_2, tmatA, tmatB

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class SACGaussianPointNetPolicy(SACGaussianPolicyBase):
    def __init__(self, in_channels=3, action_dim=5):
        super().__init__()
        self.backbone = PointNetBackbone(in_channels=in_channels, global_dim=128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x, tmatA, tmatB = self.backbone(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, tmatA, tmatB
