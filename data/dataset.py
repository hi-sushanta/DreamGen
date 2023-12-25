import torch
from torchvision import transforms
from torch import nn
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

def rescale(x, old_range,new_range,clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min)/(old_max - old_min)

    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x

class CustomDataset(Dataset):
    def __init__(self,base_dir, image_with_text,emb_size=512,target_imgsize=(64,64)):
        self.image_with_text = image_with_text
        self.target_imgsize = target_imgsize
        self.emb_size = emb_size
        self.base_dir = base_dir
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.zero_grad = False



        self.transform = transforms.Compose([
        transforms.Lambda(lambda t: rescale(t,(0,255),(-1,1))),
        transforms.Lambda(lambda x: x.permute(2,0,1))

        ])
    def __len__(self):
        return len(self.image_with_text)
    def __getitem__(self,idx):
        image_path = self.base_dir + self.image_with_text[idx][0]
        src_text = self.image_with_text[idx][1]
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.target_imgsize)
        # image = image.unsqueeze(0)
        image = np.array(image)
        image = torch.as_tensor(image, dtype=torch.float32)
        image = self.transform(image)
        src_text = src_text[:77]
        text_token = clip.tokenize(src_text).to(device)
        with torch.no_grad():
            text_emb = self.model.encode_text(text_token)

        return image,text_emb,src_text
    