# WassersteinGAN-PyTorch
## Overview
This repository contains an Pytorch implementation of WGAN, WGAN-GP, WGAN-DIV and original GAN loss function.

## About WGAN
If you're new to WassersteinGAN, here's an abstract straight from the paper[1]:

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

## Dataset 
- MNIST
- FashionMNIST
- Cifar10

## Implement
```
usage: main.py [-h] [--model {gan,dcgan}]
               [--adv_loss {wgan-gp,gan,wgan-div,wgan}] [--img_size IMG_SIZE]
               [--channels CHANNELS] [--g_num G_NUM] [--z_dim Z_DIM]
               [--g_conv_dim G_CONV_DIM] [--d_conv_dim D_CONV_DIM]
               [--lambda_gp LAMBDA_GP] [--version VERSION]
               [--lambda_aux LAMBDA_AUX] [--clip_value CLIP_VALUE]
               [--epochs EPOCHS] [--d_iters D_ITERS] [--batch_size BATCH_SIZE]
               [--num_workers NUM_WORKERS] [--g_lr G_LR] [--d_lr D_LR]
               [--lr_decay LR_DECAY] [--beta1 BETA1] [--beta2 BETA2]
               [--n_classes N_CLASSES] [--pretrained_model PRETRAINED_MODEL]
               [--train TRAIN] [--parallel PARALLEL]
               [--dataset {mnist,cifar10,fashion}]
               [--use_tensorboard USE_TENSORBOARD] [--dataroot DATAROOT]
               [--log_path LOG_PATH] [--model_save_path MODEL_SAVE_PATH]
               [--sample_path SAMPLE_PATH] [--attn_path ATTN_PATH]
               [--log_step LOG_STEP] [--sample_step SAMPLE_STEP]
               [--model_save_step MODEL_SAVE_STEP]

optional arguments:
  -h, --help            show this help message and exit
  --model {gan,dcgan}
  --adv_loss {wgan-gp,gan,wgan-div,wgan}
  --img_size IMG_SIZE
  --channels CHANNELS   number of image channels
  --g_num G_NUM         train the generator every 5 steps
  --z_dim Z_DIM         noise dim
  --g_conv_dim G_CONV_DIM
  --d_conv_dim D_CONV_DIM
  --lambda_gp LAMBDA_GP
                        for wgan gp
  --version VERSION
  --lambda_aux LAMBDA_AUX
                        aux loss number
  --clip_value CLIP_VALUE
                        lower and upper clip value for disc. weights
  --epochs EPOCHS       numer of epochs of training
  --d_iters D_ITERS
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --g_lr G_LR           use TTUR lr rate for Adam
  --d_lr D_LR           use TTUR lr rate for Adam
  --lr_decay LR_DECAY
  --beta1 BETA1
  --beta2 BETA2
  --n_classes N_CLASSES
                        how many labels in dataset
  --pretrained_model PRETRAINED_MODEL
  --train TRAIN
  --parallel PARALLEL
  --dataset {mnist,cifar10,fashion}
  --use_tensorboard USE_TENSORBOARD
  --dataroot DATAROOT
  --log_path LOG_PATH
  --model_save_path MODEL_SAVE_PATH
  --sample_path SAMPLE_PATH
  --attn_path ATTN_PATH
  --log_step LOG_STEP
  --sample_step SAMPLE_STEP
  --model_save_step MODEL_SAVE_STEP
```

## Reference
1. [WGAN](https://arxiv.org/abs/1701.07875)
2. [WGAN-GP](https://arxiv.org/abs/1704.00028)
3. [WGAN-DIV](https://arxiv.org/abs/1712.01026)
4. [DCGAN](https://arxiv.org/abs/1511.06434)
5. [CT-GAN](https://arxiv.org/abs/1803.01541)(todo)