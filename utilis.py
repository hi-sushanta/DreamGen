import torch
import torchvision
from torchvision import transforms
from transformers import T5Tokenizer
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.dataset import CustomDataset
import pandas as pd
from model.Network import Generator,Discriminator
from torch import optim 
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

class Utils():
    def __init__(self):
        pass
    def gat_dataloader(self,base_dir,data_filepath,target_size):
        pd_df = pd.read_excel(data_filepath)
        pd_df = pd_df.dropna()
        pd_df.drop("comment_number",axis=1,inplace=True)
        image_with_text = pd_df.to_numpy().squeeze()
        dataset = CustomDataset(base_dir,image_with_text,target_size)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=4)
        return dataloader

    def get_model(self,in_channels,n_feat,embedding_dim,
                        image_size, learning_rate, beta_value=(0.9,0.999),
                        device='cpu', if_pretrain=False,gan_path=None):
        
        gan = nn.DataParallel(Generator(in_channels,n_feat,embedding_dim,image_size).to(device))

        # Create the optimizer with Adam (or any other optimizer you prefer)
        g_optim = optim.Adam(gan.parameters(), lr=learning_rate,betas=beta_value)
        

        if if_pretrain:
            print("Using pretrained model")
            gan_statedict = torch.load(gan_path)
            gan.load_state_dict(gan_statedict)
        
        return gan,g_optim
    

   