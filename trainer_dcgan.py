# %% 
"""
wgan with different loss function, used the pure dcgan structure.
"""
import os 
import time
import torch
import datetime

import torch.nn as nn 
import torchvision
from torchvision.utils import save_image

import numpy as np

import sys 
sys.path.append('.')
sys.path.append('..')

from models.dcgan import Generator, Discriminator
from utils.utils import *

# %%
class Trainer_dcgan(object):
    def __init__(self, data_loader, config):
        super(Trainer_dcgan, self).__init__()

        # data loader 
        self.data_loader = data_loader

        # exact model and loss 
        self.model = config.model
        self.adv_loss = config.adv_loss

        # model hyper-parameters
        self.imsize = config.img_size 
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.channels = config.channels
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.epochs = config.epochs
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers 
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr 
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.n_classes = config.n_classes
        self.lambda_aux = config.lambda_aux
        self.clip_value = config.clip_value

        self.dataset = config.dataset 
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.dataroot 
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.version = config.version

        # path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)

        if self.use_tensorboard:
            self.build_tensorboard()

        self.build_model()

    def train(self):
        '''
        Training
        '''

        real_label = 0.9
        fake_label = 0.0

        # for gan loss to value
        valid = tensor2var(torch.full((self.batch_size,), real_label))
        fake = tensor2var(torch.full((self.batch_size,), fake_label))
        # fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim)) # （*, 100）

        for epoch in range(self.epochs):
            # start time
            start_time = time.time()

            for i, (real_images, labels) in enumerate(self.data_loader):

                # configure input 
                if self.adv_loss == 'wgan-div':
                    real_images = tensor2var(real_images, grad=True) # for wgan div to compute grad
                else:
                    real_images = tensor2var(real_images)
                
                # ==================== Train D ==================
                self.D.train()
                self.G.train()

                self.D.zero_grad()

                # compute loss with real images 
                d_out_real = self.D(real_images)

                if self.adv_loss == 'wgan-gp' or self.adv_loss == 'wgan-div' or self.adv_loss == 'wgan':
                    d_loss_real = - torch.mean(d_out_real)
                elif self.adv_loss == 'gan':
                    d_loss_real = self.adversarial_loss_sigmoid(d_out_real, valid)

                # noise z for generator
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim)) # 64, 100

                fake_images = self.G(z)
                d_out_fake = self.D(fake_images)

                if self.adv_loss == 'wgan-gp' or self.adv_loss == 'wgan-div' or self.adv_loss == 'wgan':
                    d_loss_fake = torch.mean(d_out_fake)
                elif self.adv_loss == 'gan':
                    d_loss_fake = self.adversarial_loss_sigmoid(d_out_fake, fake)

                # total d loss
                d_loss = d_loss_real + d_loss_fake

                # for the wgan loss function
                if self.adv_loss == 'wgan-gp':
                    grad = compute_gradient_penalty(self.D, real_images, fake_images)
                    d_loss = self.lambda_gp * grad + d_loss
                elif self.adv_loss == 'wgan-div':
                    grad = compute_gradient_penalty_div(d_out_real, d_out_fake, real_images, fake_images)
                    d_loss = d_loss + grad

                d_loss.backward()
                # update D
                self.d_optimizer.step()

                if self.adv_loss == 'wgan':
                    # clip weights of discriminator
                    for p in self.D.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                # train the generator every 5 steps
                if i % self.g_num == 0:

                    # =================== Train G and gumbel =====================
                    self.G.zero_grad()
                    # create random noise 
                    fake_images = self.G(z)

                    # compute loss with fake images 
                    g_out_fake = self.D(fake_images) # batch x n

                    if self.adv_loss == 'wgan-gp' or self.adv_loss == 'wgan-div' or self.adv_loss == 'wgan':
                        g_loss_fake = - torch.mean(g_out_fake)
                    elif self.adv_loss == 'gan':
                        g_loss_fake = self.adversarial_loss_sigmoid(g_out_fake, valid)

                    g_loss_fake.backward()
                    # update G
                    self.g_optimizer.step()

            # log to the tensorboard
            self.logger.add_scalar('d_loss', d_loss.data, epoch)
            self.logger.add_scalar('g_loss_fake', g_loss_fake.data, epoch)
            # end one epoch

            # print out log info
            if (epoch) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_gp: {:.4f}, g_loss: {:.4f}, "
                    .format(elapsed, epoch, self.epochs, epoch,
                            self.epochs, d_loss.item(), g_loss_fake.item()))

            # sample images 
            if (epoch) % self.sample_step == 0:
                self.G.eval()
                # save real image
                save_sample(self.sample_path + '/real_images/', real_images, epoch)
                
                with torch.no_grad():
                    fake_images = self.G(fixed_z)
                    # save fake image 
                    save_sample(self.sample_path + '/fake_images/', fake_images, epoch)
                    
                # sample sample one images
                # self.number_real, self.number_fake = save_sample_one_image(self.G, self.sample_path, real_images, epoch, z_dim=self.z_dim, n_classes=self.n_classes)


    def build_model(self):

        self.G = Generator(batch_size = self.batch_size, image_size = self.imsize, z_dim = self.z_dim, conv_dim = self.g_conv_dim, channels = self.channels, n_classes=self.n_classes).cuda()
        self.D = Discriminator(batch_size = self.batch_size, image_size = self.imsize, conv_dim = self.d_conv_dim, channels = self.channels, n_classes=self.n_classes).cuda()

        # apply the weights_init to randomly initialize all weights
        # to mean=0, stdev=0.2
        self.G.apply(init_weight)
        self.D.apply(init_weight)
        
        # optimizer 
        if self.adv_loss == 'wgan':
            # the original wgan use the RMSprop optimizer.
            self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), self.g_lr)
            self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), self.d_lr)
        else:
            self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
            self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # for orignal gan loss function
        self.adversarial_loss_sigmoid = nn.BCEWithLogitsLoss()

        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def save_image_tensorboard(self, images, text, step):
        if step % 100 == 0:
            img_grid = torchvision.utils.make_grid(images, nrow=8)

            self.logger.add_image(text + str(step), img_grid, step)
            self.logger.close()
