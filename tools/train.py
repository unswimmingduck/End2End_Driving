from e2e_cnn.dataset.gym_dockey import Dockey_Dataset
from e2e_cnn.model.e2e_cnn import e2e_net

from e2e_cnn.util.logger import get_root_logger
from e2e_cnn.util.step_lr import lr_stepping
from e2e_cnn.util.optim import build_optimizer
from e2e_cnn.util.utils import AverageMeter, checkpoint_save

from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

import argparse
import yaml
from munch import Munch

import os.path as osp
import os

import time
import datetime
from tqdm import tqdm


def main():
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    writer = SummaryWriter(cfg.work_dir)

    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dir', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')


    model = e2e_net().cuda()
    data_config = cfg.data
    train_dataset = Dockey_Dataset(data_config.train.info_path)
    train_dataloader = DataLoader(
                                  train_dataset, 
                                  batch_size=data_config.train.batch_size, num_workers=data_config.train.num_workers,
                                  shuffle=data_config.train.shuffle, drop_last=True)


    val_dataset = Dockey_Dataset(data_config.val.info_path)
    val_dataloader = DataLoader(
                                val_dataset, 
                                batch_size=data_config.val.batch_size, num_workers=data_config.val.num_workers,
                                shuffle=data_config.val.shuffle, drop_last=True)

    optimizer = build_optimizer(model, cfg.optimizer)

    start_epoch = 1
    if cfg.checkpoint is not None:
        model, start_epoch, optimizer = load_checkpoint(cfg.checkpoint, model, start_epoch, optimizer) 

    
    for i in range(start_epoch, cfg.epoch+1):
        train(model, i, optimizer, logger, train_dataloader, cfg, writer)
        if i%cfg.val_epoch == 0:
            validate(model, i, logger, val_dataloader, data_config.val, writer)
    writer.flush()



def get_args():
    parser = argparse.ArgumentParser('E2E_CNN')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--work_dir', type=str, help='work dir path')
    args = parser.parse_args()
    return args

def load_checkpoint(path, model, start_epoch, optimizer):
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return  model, start_epoch, optimizer


def train(model, epoch, optimizer, logger, dataloader, cfg, writer):
    data_time = AverageMeter()
    iter_time = AverageMeter()
    start_time = time.time()
    
    lr_stepping(optimizer, epoch, cfg.step_epoch, cfg.epoch, optimizer.param_groups[0]['lr'])
    lr = optimizer.param_groups[0]['lr']

    for i, (img, angle_label) in enumerate(dataloader, start=1):
        data_time.update(time.time() - start_time)
           
        loss = model(img, angle_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_time.update(time.time() - start_time)
        remain_iter = len(dataloader) * (cfg.epoch - epoch + 1) - i
        remain_time = remain_iter * (iter_time.avg + data_time.avg)
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))

        logger.info(f'Epoch [{epoch}/{cfg.epoch}][{i}/{len(dataloader)}], lr:{lr:.7f}, eta: {remain_time}, data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}, loss: {loss:.5f}')
        start_time = time.time()
        del img, angle_label

    checkpoint_save(epoch, model, optimizer, cfg.work_dir)

    writer.add_scalar("train/loss", loss, epoch)
    writer.add_scalar("train/lr", lr, epoch)


def validate(model, epoch, logger, val_dataloader, cfg, writer):
    logger.info('Validation Start')
    loss = AverageMeter()
    val_bar = tqdm(total=len(val_dataloader)) 
    for i, (img, angle_label) in enumerate(val_dataloader, start=1):
        loss.update(model(img, angle_label))
        val_bar.update(cfg.batch_size)
    del img, angle_label
    writer.add_scalar("val/loss", loss.avg, i)
    logger.info('Validation result loss: {}'.format(loss.avg))


if __name__ == '__main__':
    main()