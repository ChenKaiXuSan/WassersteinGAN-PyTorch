# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import numpy as np

from utils.utils import init_weight

# %%
def fill_labels(img_size):
    '''
    for D fill labels

    Args:
        img_size (int): image size

    Returns:
        tensor: filled type
    '''    
    fill = torch.zeros([10, 10, img_size, img_size])
    for i in range(10):
        fill[i, i, :, :] = 1
    return fill.cuda()

# %%
class Generator(nn.Module):
    '''
    Generator

    '''
    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, channels = 1, n_classes = 10):
        
        super(Generator, self).__init__()
        self.imsize = image_size
        self.n_classes = n_classes
        self.channels = channels
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(n_classes, n_classes).cuda()

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8

        self.fc1 = nn.Linear(self.n_classes + self.z_dim, self.n_classes + self.z_dim)

        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(self.n_classes + self.z_dim, conv_dim * mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )
        
        curr_dim = int(curr_dim / 2)
        
        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):

        # labels_emb = self.label_emb(labels).cuda() # (*, 10)
        
        labels_one_hot = F.one_hot(labels, num_classes=self.n_classes) # (*, 10)

        input = torch.cat((labels_one_hot, z), 1) # (*, 110)

        input = self.fc1(input) # (*, 110)

        # input = input.unsqueeze(2).unsqueeze(3)
        input = input.view(input.size(0), -1, 1, 1) # (*, 110, 1, 1)

        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        out = self.last(out)

        return out


# %%
class Discriminator(nn.Module):
    '''
    discriminator, Auxiliary classifier

    '''
    def __init__(self, batch_size, n_classes = 10, image_size = 64, conv_dim = 64, channels = 1):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.channels = channels
        self.n_classes = n_classes

        self.label_embedding = nn.Embedding(self.n_classes, self.n_classes).apply(init_weight)
        
        self.l1 = nn.Sequential(
            nn.Conv2d(self.channels, conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1)
        )

        curr_dim = conv_dim

        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2, 0.8),
            nn.LeakyReLU(0.1)
        )

        curr_dim = curr_dim * 2

        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2, 0.8),
            nn.LeakyReLU(0.1)
        )
        
        curr_dim = curr_dim * 2

        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2, 0.8),
            nn.LeakyReLU(0.1)
        )

        curr_dim = curr_dim * 2
        
        # output layers
        self.last_adv = nn.Sequential(
            nn.Linear(curr_dim, 1),
            # nn.Sigmoid()
            )
        
        self.embed = nn.Sequential(
            nn.Embedding(self.n_classes, curr_dim)
            )

        self.last_aux = nn.Sequential(
            nn.Linear(curr_dim , self.n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, labels):
        batch_size, c, h, w = x.size()
        # labels_fill = fill_labels(x.size()[2])[labels] # 10, 10, img_size, img_size torch.cuda.floattensor
        # input = torch.cat((x, labels_fill), 1) # torch.cuda.floattensor

        out = self.l1(x) # (*, 64, 32, 32)
        out = self.l2(out) # (*, 128, 16, 16)
        out = self.l3(out) # (*, 256, 8, 8)
        out = self.l4(out) # (*, 512, 4, 4)
        
        out = F.leaky_relu(out, 0.2)
        out = torch.sum(out, dim=(2, 3)) # global sumpool (*, 512)

        validity = self.last_adv(out) # (*, 1)
        
        if labels is not None:
            validity += torch.sum(self.embed(labels)*out, dim=1, keepdim=True)

        label = self.last_aux(out) # (*, 10)

        return validity, label

# %% 
if __name__ == '__main__':
    genertor = Generator(64, image_size=64).cuda()
    discriminator = Discriminator(64).cuda()

    # print(genertor, discriminator)

    image = torch.randn(64, 1, 64, 64)
    image = image.cuda()

    z = torch.randn(64, 100)
    z = z.cuda()

    label = torch.LongTensor(np.random.randint(0, 10, image.size()[0])).cuda()

    y, p1 = discriminator(image, label)

    x, att1 = genertor(z, label)
    print(x.shape, att1.shape)
