'''Train MNIST model'''
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

from lib.custom_layers import *
from lib.dataset_utils import *
from lib.mnist_model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def loss_function(z_tuple, targets, k):
    """Loss function used in Gowal et al."""
    z, z_ub, z_lb = z_tuple
    loss = F.cross_entropy(z, targets, reduction='sum')
    eye = torch.eye(z.size(1), device=z.device)
    mask = eye[targets]
    z_hat = (1 - mask) * z_ub + mask * z_lb
    loss_ub = F.cross_entropy(z_hat, targets, reduction='sum')
    return k * loss + (1 - k) * loss_ub, z_hat


def evaluate(net, dataloader, k, params, device):

    net.eval()
    val_loss = 0
    correct = 0
    ibp_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            z_tuple = net(inputs, params)
            loss, z_hat = loss_function(z_tuple, targets, k)
            val_loss += loss.item()
            _, predicted = z_tuple[0].max(1)
            _, ibp_predicted = z_hat.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            ibp_correct += ibp_predicted.eq(targets).sum().item()

    return val_loss / total, ibp_correct / total, correct / total


def train(net, trainloader, validloader, optimizer, epoch, k, params, device,
          log, save_best_only=True, best_acc=0, model_path='./model.pt',
          save_after_epoch=0):

    net.train()
    train_loss = 0
    correct = 0
    ibp_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        z_tuple = net(inputs, params)
        loss, z_hat = loss_function(z_tuple, targets, k)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = z_tuple[0].max(1)
        _, ibp_predicted = z_hat.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        ibp_correct += ibp_predicted.eq(targets).sum().item()

    val_loss, val_ibp_acc, val_acc = evaluate(
        net, validloader, k, params, device)

    log.info(' %5d | %.4f, %.4f, %.4f | %8.4f, %9.4f, %7.4f', epoch,
             train_loss / total, ibp_correct / total, correct / total,
             val_loss, val_ibp_acc, val_acc)

    # Save model weights
    if epoch >= save_after_epoch:
        if save_best_only and val_ibp_acc > best_acc:
            log.info('Saving model...')
            torch.save(net.state_dict(), model_path + '.h5')
            best_acc = val_ibp_acc
        elif not save_best_only:
            log.info('Saving model...')
            torch.save(net.state_dict(), model_path + '_epoch%d.h5' % epoch)
    return best_acc


def main():

    # Set experiment id
    exp_id = 12
    model_name = 'mnist_linf_ibp_exp%d' % exp_id

    # Training parameters
    batch_size = 100
    epochs = 150
    data_augmentation = False
    learning_rate = 1e-3
    l1_reg = 0
    l2_reg = 0

    # IBP parameters
    k_init = 1
    k_final = 0.5
    k_warmup_epoch = 40  # "warm-up" Original: 4
    eps_init = 0
    eps_final = 0.5
    eps_warmup_epoch = 40  # "ramp-up" Original: 20

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Set all random seeds
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    # Get logger
    log_file = model_name + '.log'
    log = logging.getLogger('train_mnist')
    log.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(name)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    log.info(log_file)
    log.info(('MNIST | exp_id: {}, seed: {}, init_learning_rate: {}, ' +
              'batch_size: {}, l2_reg: {}, l1_reg: {}, epochs: {}, ' +
              'data_augmentation: {}, subtract_pixel_mean: {}').format(
                  exp_id, seed, learning_rate, batch_size, l2_reg, l1_reg,
                  epochs, data_augmentation, subtract_pixel_mean))

    log.info('Preparing data...')
    # trainloader, validloader, testloader = load_mnist(
    #     batch_size, data_dir='/data', val_size=0.1, shuffle=True, seed=seed)
    trainloader, _, validloader = load_mnist(
        batch_size, data_dir='/data', val_size=0., shuffle=True, seed=seed)

    log.info('Building model...')
    # params = {'p': 2,
    #           'epsilon': eps_init,
    #           'input_bound': (0, 1)}
    # TODO: change LpLinear to your layer
    # net = IBPSmallCustom(LpLinear, params)
    params = {'epsilon': eps_init,
              'input_bound': (0, 1)}
    net = IBPSmallCustom(CardLinear, params)
    # net = IBPMedium()
    # net = IBPBasic()
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [30, 50], gamma=0.1)

    log.info(' epoch |   loss,  b-acc,    acc | val_loss, val_b-acc, val_acc')
    best_acc = 0
    for epoch in range(epochs):

        # linear schedule on k
        if epoch < k_warmup_epoch:
            k = (k_final - k_init) * epoch / k_warmup_epoch + k_init
        else:
            k = k_final
        # if epoch < 10:
        #     k = k_init
        # elif epoch < k_warmup_epoch:
        #     k = (k_final - k_init) * (epoch - 10) / \
        #         (k_warmup_epoch - 10) + k_init
        # else:
        #     k = k_final

        # linear schedule on epsilon
        if epoch < eps_warmup_epoch:
            eps = (eps_final - eps_init) * epoch / eps_warmup_epoch + eps_init
        else:
            eps = eps_final
        # if epoch < 10:
        #     eps = 0
        # elif epoch < eps_warmup_epoch:
        #     eps = (eps_final - eps_init) * (epoch - 10) / \
        #         (eps_warmup_epoch - 10) + eps_init
        # else:
        #     eps = eps_final

        params['epsilon'] = eps
        best_acc = train(net, trainloader, validloader, optimizer, epoch,
                         k, params, device, log, save_best_only=True,
                         best_acc=best_acc, model_path=model_path,
                         save_after_epoch=eps_warmup_epoch)
        # lr_scheduler.step()

    # loss, ibp_acc, acc = evaluate(net, testloader, k, eps, device)
    # log.info('Test loss: %.4f, Worst-case test acc: %.4f, Test acc: %.4f',
    #          loss, ibp_acc, acc)


if __name__ == '__main__':
    main()
