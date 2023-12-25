import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import math 
import torch.nn.functional  as F

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        ]

        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        out = self.model(x)
        return out

class Perior(nn.Module):

    def __init__(self,):
        super(Perior,self).__init__()
        # self.model, self.preprocess = clip.load("ViT-B/32", device='cpu')
        # self.model.zero_grad = False
        self.multihead = MultiheadAttention(512,512,512)
        self.li1 = nn.Linear(512,512)
        self.li2 = nn.Linear(512,512)


    def forward(self,c):
        last_output = self.multihead(c.unsqueeze(dim=0))
        last_output = self.li1(last_output)
        last_output = self.li2(last_output)
        return last_output.squeeze()

class ResNetBlock(nn.Module):
    def __init__(self,in_channels,channels):
        super(ResNetBlock,self).__init__()
        self.gn = nn.GroupNorm(channels,channels)
        self.sact = nn.SiLU()
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=(1,1))

    def forward(self,x):
        temp = self.conv2(x)
        x = self.sact(self.gn(x))
        x = self.conv1(x)
        x = self.sact(self.gn(x))
        x = self.conv1(x)
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
    def __init__(self,channels,out_channels,atten_dim = 64,kernel_size=(3,3),stride=2,num_resblock=2,attention=False):
        super(DBBlock,self).__init__()
        self.is_atten = attention
        self.conv1 = nn.Conv2d(channels,out_channels,kernel_size=kernel_size,stride=stride)
        self.res = ResNetBlock(out_channels,out_channels)
        self.attention = SelfAttention(out_channels)#,output_size=43*2)
        self.num_resblock = num_resblock

    def forward(self,x):
        x = self.conv1(x)
        # x = x + emb_value
        # print(x.shape)
        for i in range(self.num_resblock):
            x = self.res(x)
        if self.is_atten:
            x = self.attention(x)
        return x

class UBBlock(nn.Module):
    def __init__(self,channels,out_channel,kernel_size=(3,3),stride=2,num_resblock=1,is_atten=False,if_skip=True):
        super(UBBlock,self).__init__()
        self.is_atten = is_atten
        self.tconv = nn.ConvTranspose2d(channels,out_channel,kernel_size=kernel_size,stride=stride)

        self.res = ResNetBlock(channels,channels)
        self.attention = SelfAttention(channels)
        self.num_resblock = num_resblock
        self.if_skip=if_skip
    def forward(self,x,skip=None):
        if skip is not None:
            x = torch.cat([x,skip],dim=1)

        for i in range(self.num_resblock):
            x = self.res(x)
        if self.is_atten:
            x = self.attention(x.squeeze())
        x = self.tconv(x)
        return x


def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0,1))
    xmin = x.min((0,1))
    return(x - xmin)/(xmax - xmin)

def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t,s] = unorm(store[t,s])
    return nstore

def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2,3))
    xmin = x.min((2,3))
    xmax = np.expand_dims(xmax,(2,3))
    xmin = np.expand_dims(xmin,(2,3))
    nstore = (x - xmin)/(xmax - xmin)
    return torch.from_numpy(nstore)


def plot_grid(x,n_sample,n_rows,save_dir,w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample//n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print('saved image at ' + save_dir + f"run_image_w{w}.png")
    return grid

def plot_sample(x_gen_store,n_sample,nrows,save_dir, fn,  w, save=False):
    ncols = n_sample//nrows
    sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow

    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
        return plots
    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0])
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
    return ani
    
class Generator(nn.Module):
    def __init__(self,input_channel,output_channel,emb_size,img_size):
        super(Generator,self).__init__()

        self.perior = Perior()
        self.timeemb1 = EmbedFC(1,512)
        self.timeemb2 = EmbedFC(1,512)
        self.timeemb3 = EmbedFC(1,512)
        self.timeemb4 = EmbedFC(1,256)
        self.timeemb5 = EmbedFC(1,256)

        self.contexemb1 = EmbedFC(emb_size,512)
        self.contexemb2 = EmbedFC(emb_size,512)
        self.contexemb3 = EmbedFC(emb_size,512)
        self.contexemb4 = EmbedFC(emb_size,256)
        self.contexemb5 = EmbedFC(emb_size,256)
        # self.mapping_layer = MappingLayers(833,1666,833)
        self.act = nn.SiLU()
        self.n_feat = 512

        # # # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.res = ResNetBlock(input_channel,input_channel)

        self.dblock1 = DBBlock(input_channel,256,kernel_size=3,stride=2,num_resblock=3)
        self.dblock2 = DBBlock(256,256,kernel_size=3,stride=2,num_resblock=3,attention=True)
        self.dblock3 = DBBlock(256,512,kernel_size=3,stride=2,num_resblock=3)
        self.dblock4 = DBBlock(512,512,kernel_size=3,stride=2,num_resblock=6,attention=True)
        self.dblock5 = DBBlock(512,512,kernel_size=1,stride=1,num_resblock=6)

        self.to_vec = nn.Sequential(nn.AvgPool2d(2), nn.GELU())

        self.ublock1 = UBBlock(512,512,kernel_size=(3,3),stride=2,num_resblock=6,is_atten=False,if_skip=False)
        self.ublock2 = UBBlock(1024,512,kernel_size=(3,3),stride=2,num_resblock=6,is_atten=True)
        self.ublock3 = UBBlock(1024,256,kernel_size=(3,3),stride=2,num_resblock=3,is_atten=True)
        self.ublock4 = UBBlock(512,256,kernel_size=(3,3),stride=2,num_resblock=3,is_atten=True)
        self.ublock5 = UBBlock(512,256,kernel_size=(4,4),stride=2,num_resblock=3,is_atten=True)

        self.out = nn.Sequential(
            nn.Conv2d(259, 256, 3, 1, 1), 
            nn.GroupNorm(8, 256), 
            nn.SiLU(),
            nn.Conv2d(256, 3, 3, 1, 1)
        )
    def forward(self,x,text_emb,t):
        # Downsample block is start
        x = self.res(x)
        emb = self.perior(text_emb)
        emb_value = emb.view(-1)
        db1 = self.dblock1(x)
        db2 = self.dblock2(db1)
        db3 = self.dblock3(db2)
        db4 = self.dblock4(db3)
        db5 = self.dblock5(db4)
        vec = self.to_vec(db5)
        # Downsample block is end
        time_emb1 = self.timeemb1(t).view(-1,self.n_feat,1,1)
        emb_value1 = self.contexemb1(emb_value).view(-1,self.n_feat,1,1) #+ time_emb1

        emb_value1 = emb_value.view(-1,self.n_feat,1,1)
        time_emb2 = self.timeemb2(t).view(-1,self.n_feat,1,1)
        emb_value2 = self.contexemb2(emb_value).view(-1,self.n_feat,1,1)# + time_emb2
        time_emb3 = self.timeemb3(t).view(-1,self.n_feat,1,1)
        emb_value3 = self.contexemb3(emb_value).view(-1,self.n_feat,1,1) #+ time_emb3
        time_emb4 = self.timeemb4(t).view(-1,self.n_feat//2,1,1)
        emb_value4 = self.contexemb4(emb_value).view(-1,self.n_feat//2,1,1)
        time_emb5 = self.timeemb5(t).view(-1,self.n_feat//2,1,1)
        emb_value5 = self.contexemb5(emb_value).view(-1,self.n_feat//2,1,1)
        
        # Upsample block is start
        ub1 = self.act(self.ublock1(vec*emb_value1+time_emb1))
        ub2 = self.ublock2(ub1*emb_value2+time_emb2,db4)
        ub3 = self.ublock3(ub2*emb_value3+time_emb3,db3)
        ub4 = self.ublock4(ub3*emb_value4+time_emb4,db2)
        ub5 = self.ublock5(ub4*emb_value5+time_emb5,db1)
        output = self.out(torch.cat((ub5,x),1))

        return output

