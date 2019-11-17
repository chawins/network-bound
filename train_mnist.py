'''Train MNIST model'''
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

from lib.dataset_utils import *
from lib.mnist_model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_input_bound(x, eps):
    x_ub = torch.clamp(x + eps, 0, 1)
    x_lb = torch.clamp(x - eps, 0, 1)
    return x, x_ub, x_lb


def loss_function(z_tuple, targets, k):
    z, z_ub, z_lb = z_tuple
    loss = F.cross_entropy(z, targets, reduction='mean')
    eye = torch.eye(z.size(1), device=z.device)
    mask = eye[targets]
    z_hat = (1 - mask) * z_ub + mask * z_lb
    loss_ub = F.cross_entropy(z_hat, targets, reduction='mean')
    return k * loss + (1 - k) * loss_ub


def evaluate(net, dataloader, k, eps, device):

    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            x_tuple = get_input_bound(inputs, eps)
            z_tuple = net(x_tuple)
            loss = loss_function(z_tuple, targets, k)
            val_loss += loss.item()
            _, predicted = z_tuple[0].max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    return val_loss / val_total, val_correct / val_total


def train(net, trainloader, validloader, optimizer, epoch, k, eps, device,
          log, save_best_only=True, best_acc=0, model_path='./model.pt'):

    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x_tuple = get_input_bound(inputs, eps)
        z_tuple = net(x_tuple)
        loss = loss_function(z_tuple, targets, k)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = z_tuple[0].max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    val_loss, val_acc = evaluate(net, validloader, k, eps, device)

    log.info(' %5d | %.4f, %.4f | %8.4f, %7.4f', epoch,
             train_loss / train_total, train_correct / train_total,
             val_loss, val_acc)

    # Save model weights
    if not save_best_only or (save_best_only and val_acc > best_acc):
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path)
        best_acc = val_acc
    return best_acc


def main():

    # Set experiment id
    exp_id = 1
    model_name = 'mnist_linf_ibp_exp%d' % exp_id

    # Training parameters
    batch_size = 100
    epochs = 100
    data_augmentation = False
    learning_rate = 1e-3
    l1_reg = 0
    l2_reg = 0

    # IBP parameters
    k_init = 1
    k_final = 0.01
    k_warmup_epoch = 20
    eps_init = 0
    eps_final = 0.1
    eps_warmup_epoch = 20

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
    model_path = os.path.join(save_dir, model_name + '.h5')

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
    trainloader, validloader, testloader = load_mnist(
        batch_size, data_dir='/data', val_size=0.1, shuffle=True, seed=seed)

    log.info('Building model...')
    # net = IBPMedium()
    net = IBPBasic()
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    log.info(' epoch | loss  , acc    | val_loss, val_acc')
    best_acc = 0
    for epoch in range(epochs):

        if epoch < k_warmup_epoch:
            k = (k_final - k_init) * epoch / k_warmup_epoch + k_init
        else:
            k = k_final

        if epoch < eps_warmup_epoch:
            eps = (eps_final - eps_init) * epoch / eps_warmup_epoch + eps_init
        else:
            eps = eps_final

        best_acc = train(net, trainloader, validloader, optimizer, epoch,
                         k, eps, device, log, save_best_only=True,
                         best_acc=best_acc, model_path=model_path)

    test_loss, test_acc = evaluate(net, testloader, k, eps, device)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    main()
