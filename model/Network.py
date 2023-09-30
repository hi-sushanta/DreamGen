
import torch
import torchvision
from torch import nn

# Create New Version of ResNet-Block
class ResNetBlock(nn.Module):
    def __init__(self,channels):
        super(ResNetBlock,self).__init__()
        self.gn = nn.GroupNorm(1,channels)
        self.sact = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=(3,3),stride=1,padding=1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=(1,1),stride=1)

    def forward(self,x):
        temp = self.conv2(x)
        x = self.sact(self.gn(x))
        x = self.conv1(x)
        x = self.sact(self.gn(x))
        output = x + temp
        return output


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()

        #conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2) #sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x) #BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x) #BxC'xHxW
        h_projection = self.conv_h(x) #BxCxHxW

        f_projection = torch.transpose(f_projection.view(B,-1,H*W), 1, 2) #BxNxC', N=H*W
        g_projection = g_projection.view(B,-1,H*W) #BxC'xN
        h_projection = h_projection.view(B,-1,H*W) #BxCxN

        attention_map = torch.bmm(f_projection, g_projection) #BxNxN
        attention_map = self.softmax(attention_map) #sum_i_N (A i,j) = 1

        #sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map) #BxCxN
        out = out.view(B,C,H,W)

        out = self.gamma*out + x
        return out

class DBBlock(nn.Module):
    def __init__(self,channels,out_channels,stride,num_resblock,attention=False):
        super(DBBlock,self).__init__()
        self.is_atten = attention
        self.conv1 = nn.Conv2d(channels,out_channels,kernel_size=(3,3),stride=stride)
        self.res = ResNetBlock(out_channels+1)
        self.attention = SelfAttention(out_channels+1)#,output_size=43*2)
        self.num_resblock = num_resblock

    def forward(self,x,emb_value):
        x = self.conv1(x)
        x = torch.cat([x,emb_value],axis=1)
        for i in range(self.num_resblock):
            x = self.res(x)
        if self.is_atten:
            x = self.attention(x)
        return x

class UBBlock(nn.Module):
    def __init__(self,channels,out_channel,stride=2,num_resblock=1,is_atten=False,is_dropout=False):
        super(UBBlock,self).__init__()
        self.is_atten = is_atten
        self.tconv = nn.ConvTranspose2d(channels+1,out_channel,kernel_size=(3,3),stride=stride)
        self.res = ResNetBlock(channels+1)
        self.attention = SelfAttention(channels+1)
        self.num_resblock = num_resblock
        self.droplayer = nn.Dropout(0.4)
        self.is_dropout = is_dropout
    def forward(self,x,emb_value):
        x = torch.cat([x,emb_value],axis=1)
        for i in range(self.num_resblock):
            x = self.res(x)
        if self.is_atten:
            x = self.attention(x)
        if self.is_dropout:
            x = self.droplayer(x)
        x = self.tconv(x)
        return x

# Generator Architacture
class Generator(nn.Module):
    def __init__(self,input_channel,output_channel,emb_size,img_size):
        super(Generator,self).__init__()
        self.linear = nn.Linear(emb_size,img_size*img_size)
        self.time_embed = nn.Linear(1, emb_size)

        # self.mapping_layer = MappingLayers(833,1666,833)
        self.act = nn.ReLU(inplace=True)
        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.dblock1 = DBBlock(input_channel,256,stride=2,num_resblock=4)
        self.dblock2 = DBBlock(257,128,stride=2,num_resblock=4)
        self.dblock3 = DBBlock(129,64,stride=2,num_resblock=4)
        self.dblock4 = DBBlock(65,32,stride=2,num_resblock=4)
        self.dblock5 = DBBlock(33,16,stride=2,num_resblock=4,attention=True)

        self.ublock1 = UBBlock(17,17,stride=2,num_resblock=4)
        self.ublock2 = UBBlock(50,32,stride=2,num_resblock=4,is_dropout=True)
        self.ublock3 = UBBlock(97,64,stride=2,num_resblock=4)
        self.ublock4 = UBBlock(193,128,stride=2,num_resblock=4)
        self.ublock5 = UBBlock(385,256,stride=2,num_resblock=4,is_atten=True,is_dropout=True)
        self.conv1 = nn.Conv2d(3,3,kernel_size=3,stride=1)
        self.conv2 = nn.ConvTranspose2d(256,output_channel,kernel_size=2,stride=1) # Las output return
        self.conv3 = nn.Conv2d(1,1,kernel_size=3,stride=2)
        self.conv4 = nn.ConvTranspose2d(1,1,kernel_size=3,stride=2)
        # self.decoder = Decoder()
    def forward(self,x,emb_value,t):
        time_emb = self.time_embed(t)
        emb_value = emb_value + time_emb
        embedding  = self.linear(emb_value).view(1,1,256,256)
        embedding = self.act(self.conv3(embedding))
        # embedding = self.act(self.conv3(embedding))
        db1 = self.act(self.dblock1(x,embedding))
        embedding = self.conv3(embedding)
        db2 = self.act(self.dblock2(db1,embedding))
        embedding = self.conv3(embedding)
        db3 = self.act(self.dblock3(db2,embedding))
        embedding = self.conv3(embedding)
        db4 = self.act(self.dblock4(db3,embedding))
        embedding = self.conv3(embedding)
        db5 = self.act(self.dblock5(db4,embedding))
        # edb5 = self.mapping_layer(db5).view((1,17,7,7))
        ub1 = self.act(self.ublock1(db5,embedding))
        embedding = self.conv4(embedding)

        ub2 = self.act(self.ublock2(torch.cat([ub1,db4],dim=1),embedding))
        embedding = self.conv4(embedding)
        ub3 = self.act(self.ublock3(torch.cat([ub2,db3],dim=1),embedding))
        embedding = self.conv4(embedding)
        ub4 = self.act(self.ublock4(torch.cat([ub3,db2],dim=1),embedding))
        embedding = self.conv4(embedding)
        ub5 = self.act(self.ublock5(torch.cat([ub4,db1],dim=1),embedding))
        output = self.conv2(ub5)

        return output

# Discriminator Architacture
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    # Weight initialization
    self.conv1 = nn.Conv2d(3,64,(4,4),stride=(2,2),padding=1,bias=False)
    self.act1 = nn.LeakyReLU(negative_slope=0.2)

    self.conv2 = nn.Conv2d(64,128,(4,4),stride=(2,2),padding=1,bias=False)
    self.bnorm1 = nn.BatchNorm2d(128)
    self.act2 = nn.LeakyReLU(negative_slope=0.2)

    self.conv3 = nn.Conv2d(128,256,(4,4),stride=(2,2),padding=1,bias=False)
    self.bnorm2 = nn.BatchNorm2d(256)
    self.act3 = nn.LeakyReLU(negative_slope=0.2)

    self.conv4 = nn.Conv2d(256,512,(4,4),stride=(2,2),padding=1,bias=False)
    self.bnorm3 = nn.BatchNorm2d(512)
    self.act4 = nn.LeakyReLU(negative_slope=0.2)

    self.conv5 = nn.Conv2d(512,512,(4,4),padding=1,bias=False)
    self.bnorm4 = nn.BatchNorm2d(512)
    self.act5 = nn.LeakyReLU(negative_slope=0.2)

    self.conv6 = nn.Conv2d(512,3,(4,4),padding=1,bias=False)
    self.patch_out = nn.Sigmoid()

    # weight initializer all conv2d layer
    self._initialize_weights()
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

  def forward(self,s_img):

    # C64: 4x4 kernel stride 2x2
    x = self.act1(self.conv1(s_img))
    # C128: 4x4 kernel stride 2x2
    x = self.act2(self.bnorm1(self.conv2(x)))
    # C256: 4x4 kernel stride 2x2
    x = self.act3(self.bnorm2(self.conv3(x)))
    # C512: 4x4 kernel stride 2x2
    x = self.act4(self.bnorm3(self.conv4(x)))
    # C512: 4x4 kernel stride 2x2
    x = self.act5(self.bnorm4(self.conv5(x)))
    # Patch Output
    x = self.patch_out(self.conv6(x))
    return x

