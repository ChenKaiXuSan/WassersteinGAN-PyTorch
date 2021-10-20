# %% 
"""
wgan with different loss function
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

        valid = tensor2var(torch.full((self.batch_size, 1), real_label))
        fake = tensor2var(torch.full((self.batch_size, 1), fake_label))

        for epoch in range(self.epochs):
            # start time
            start_time = time.time()

            for i, (real_images, labels) in enumerate(self.data_loader):

                # configure input 
                if self.adv_loss == 'wgan-div':
                    real_images = tensor2var(real_images, grad=True) # for wgan div to compute grad
                else:
                    real_images = tensor2var(real_images)
                
                labels = tensor2var(labels)

                # ==================== Train D ==================
                self.D.train()
                self.G.train()

                self.D.zero_grad()

                gen_labels = tensor2var(torch.LongTensor(np.random.randint(0, self.n_classes, labels.size()[0])))

                # compute loss with real images 
                d_out_real, real_aux = self.D(real_images, labels)

                if self.adv_loss == 'wgan-gp' or self.adv_loss == 'wgan-div':
                    d_loss_real = - torch.mean(d_out_real)
                elif self.adv_loss == 'gan':
                    d_loss_real = self.adversarial_loss_sigmoid(d_out_real, valid)

                # noise z for generator
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim)) # 64, 100

                fake_images = self.G(z, labels)
                d_out_fake, fake_aux = self.D(fake_images, labels)

                if self.adv_loss == 'wgan-gp' or self.adv_loss == 'wgan-div':
                    d_loss_fake = torch.mean(d_out_fake)
                elif self.adv_loss == 'gan':
                    d_loss_fake = self.adversarial_loss_sigmoid(d_out_fake, fake)

                # total d loss
                d_loss = d_loss_real + d_loss_fake

                # for the wgan loss function
                if self.adv_loss == 'wgan-gp':
                    grad = compute_gradient_penalty(self.D, real_images, fake_images, labels)
                    d_loss = self.lambda_gp * grad + d_loss
                elif self.adv_loss == 'wgan-div':
                    grad = compute_gradient_penalty_div(d_out_real, d_out_fake, real_images, fake_images)
                    d_loss = d_loss + grad

                d_loss.backward()
                # update D
                self.d_optimizer.step()
                # log to the tensorboard
                self.logger.add_scalar('d_loss', d_loss.data, epoch)

                # train the generator every 5 steps
                if i % self.g_num == 0:

                    # =================== Train G and gumbel =====================
                    self.G.zero_grad()
                    # create random noise 
                    # z = tensor2var(torch.randn(real_images.size()[0], self.z_dim)) # (*, z_dim)
                    gen_labels = tensor2var(torch.LongTensor(np.random.randint(0, self.n_classes, labels.size()[0])))
                    fake_images = self.G(z, labels)

                    # compute loss with fake images 
                    g_out_fake, pred_labels = self.D(fake_images, labels) # batch x n

                    if self.adv_loss == 'wgan-gp' or self.adv_loss == 'wgan-div':
                        g_loss_fake = - torch.mean(g_out_fake)
                    elif self.adv_loss == 'gan':
                        g_loss_fake = self.adversarial_loss_sigmoid(g_out_fake, valid)

                    g_loss_fake.backward()
                    # update G
                    self.g_optimizer.step()

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
                self.save_sample(real_images, epoch)
                # make the fake labels by classes 
                labels = np.array([num for _ in range(self.n_classes) for num in range(self.n_classes)]) # (10, 10)
                
                # fixed input for debugging
                fixed_z = tensor2var(torch.randn(self.n_classes ** 2, self.z_dim)) # 100, 100

                with torch.no_grad():
                    labels = to_LongTensor(labels)
                    fake_images = self.G(fixed_z, labels)
                    # save fake image 
                    save_image(fake_images[:100].data, 
                                os.path.join(self.sample_path + '/fake_images/', '{}_fake.png'.format(epoch + 1)), nrow=self.n_classes, normalize=True)
                
                # sample sample one images
                self.number_real, self.number_fake = save_sample_one_image(self.G, self.sample_path, real_images, epoch, z_dim=self.z_dim, n_classes=self.n_classes)


    def build_model(self):

        self.G = Generator(batch_size = self.batch_size, image_size = self.imsize, z_dim = self.z_dim, conv_dim = self.g_conv_dim, channels = self.channels, n_classes=self.n_classes).cuda()
        self.D = Discriminator(batch_size = self.batch_size, image_size = self.imsize, conv_dim = self.d_conv_dim, channels = self.channels, n_classes=self.n_classes).cuda()

        # apply the weights_init to randomly initialize all weights
        # to mean=0, stdev=0.2
        self.G.apply(init_weight)
        self.D.apply(init_weight)
        
        # loss and optimizer 
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = nn.NLLLoss()
        # for orignal gan loss function
        self.adversarial_loss_sigmoid = nn.BCEWithLogitsLoss()

        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def save_sample(self, real_images, step):
        path = self.sample_path + '/real_images/'
        save_image(real_images.data[:100], os.path.join(path, '{}_real.png'.format(step + 1)), normalize=True, nrow=self.n_classes)

    def save_image_tensorboard(self, images, text, step):
        if step % 100 == 0:
            img_grid = torchvision.utils.make_grid(images, nrow=8)

            self.logger.add_image(text + str(step), img_grid, step)
            self.logger.close()
