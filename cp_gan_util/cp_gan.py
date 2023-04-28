from PIL import Image
import torchvision.transforms as T
import numpy as np
from torch import nn
import cv2
import torch

from object_discovery_cp_gan.cpgan_model import MyUNet


def create_mask(path, mask_size):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    G_net = MyUNet(3, 1, border_zero=True).to(device)
    G_net = nn.DataParallel(G_net)
    G_net.load_state_dict(torch.load('cpgan_G_49.pt', map_location=device))
    img = Image.fromarray(path).convert('RGB')
    transform = T.Resize(size=mask_size)
    convert_tensor = T.ToTensor()
    img = convert_tensor(transform(img))
    img = img[:3]

    img = img.clone().detach().unsqueeze(0).repeat(64, 1, 1, 1)
    result, _ = G_net(img)
    mask = np.expand_dims(np.squeeze(result.cpu().detach().numpy()).transpose((1, 2, 0)).sum(axis=-1), axis=-1)
    return mask


def reduce_size(path, new_size):
    image = np.array(Image.open(path).convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, new_size)
    return image, new_size
