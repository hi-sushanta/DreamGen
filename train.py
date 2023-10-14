from model import Network
from utilis import Utils
import torch
import torchvision
from IPython.display import clear_output
import torch.nn.functional as F
from torchvision import transforms
from transformers import T5Tokenizer
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import torch.optim as optim
from loss.loss import *

# Hyperparameter
base_dir = "/home/chi/Downloads/flicker-text-to-image/flickr30k_images/flickr30k_images/"
xls_datapath = "/home/chi/Documents/Deep-Learning-Project/Text-To-Image/results.xlsx"
# hyperparameters

# diffusion hyperparameters
timesteps = 1000
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64 # 64 hidden dimension feature
n_cfeat = 512 # context vector is of size 5
height = 64 # 16x16 image
save_dir = './weights/'

# training hyperparameters
batch_size = 100
n_epoch = 32
lrate=1e-3
# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

ut = Utils()


# Train the Model
# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


def train(epoch):
    generator,gen_opt = ut.get_model(in_channels=3,out_channels=n_feat,embedding_dim=n_cfeat,image_size=height,learning_rat=lrate)

    dataloader = ut.gat_dataloader(base_dir=base_dir,data_filepath=xls_datapath,target_size=(height,height))    

    for e in range(epoch):

        generator.train()
        
        gen_opt.param_groups[0]['lr'] = lrate*(1-e/epoch)
        
        for (image, emb_value,txt) in tqdm(dataloader):
            gen_opt.zero_grad()
            image = image.to(device)
            emb_value = emb_value.squeeze().to(device)

            # perturb data
            noise = torch.randn_like(image)
            t = torch.randint(1, timesteps + 1, (image.shape[0],)).to(device)
            x_pert = perturb_input(image, t, noise)
            pt = generator(x_pert,emb_value,t/timesteps)
            # loss is mean squared error between the predicted and true noise
            loss = ut.mse_loss(pred_noise, noise)
            total_loss.append(loss.item())
            loss.backward()
            gen_opt.step()
           
           # save model periodically
        if ep%4==0 or ep == int(n_epoch-1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(generator.state_dict(), save_dir + f"context_model_{ep}.pth")
            print('saved model at ' + save_dir + f"context_model_{ep}.pth")
            print(f"Loss>>>>>{sum(total_loss)/len(dataloader)}")

        

if __name__ == "__main__":
    # Train Diffusion Generative Model
    train(epoch)