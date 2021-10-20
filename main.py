# %%
import sys
import os

from torch.utils.data import dataloader

sys.path.append('..')
sys.path.append('.')

from trainer_dcgan import Trainer_dcgan
from utils.utils import *
from dataset.dataset import getdDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
# %%
def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='dcgan', choices=['gan', 'dcgan'])
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['wgan-gp', 'gan', 'wgan-div', 'wgan'])
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--g_num', type=int, default=5, help='train the generator every 5 steps')
    parser.add_argument('--z_dim', type=int, default=100, help='noise dim')
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10, help='for wgan gp')
    parser.add_argument('--version', type=str, default='test')
    parser.add_argument('--lambda_aux', type=int, default=0.3, help='aux loss number')
    parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')

    # Training setting
    parser.add_argument('--epochs', type=int, default=500, help='numer of epochs of training')
    parser.add_argument('--d_iters', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    # TTUR 
    parser.add_argument('--g_lr', type=float, default=0.0001, help='use TTUR lr rate for Adam')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='use TTUR lr rate for Adam')
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--n_classes', type=int, default=10, help='how many labels in dataset')

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'fashion'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--dataroot', type=str, default='../data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)


    return parser.parse_args()

# %%
def main(config):
    # data loader 
    data_loader = getdDataset(config)

    # delete the exists path
    del_folder(config.sample_path, config.version)
    del_folder(config.log_path, config.version)
    del_folder(config.sample_path, config.version + '/real_images')

    # create directories if not exist
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.sample_path, config.version + '/real_images')
    make_folder(config.sample_path, config.version + '/fake_images')

    if config.train:
        if config.model == 'dcgan':
            trainer = Trainer_dcgan(data_loader, config)
        trainer.train()
    
# %% 
if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)