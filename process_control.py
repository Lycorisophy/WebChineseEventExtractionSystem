import logging
import os
import torch
import argparse
import json
import numpy as np
import sys


def app_path():
    """Returns the base application path."""
    if hasattr(sys, 'frozen'):
        # Handles PyInstaller
        return os.path.dirname(sys.executable).replace("\\", "/")
    return os.path.dirname(__file__).replace("\\", "/")


def get_args(filename='commandline_args.txt'):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(filename, 'r') as f:
        args.__dict__ = json.load(f)
    return args


def get_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    return logging.getLogger(__name__)


def set_environ():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def label_from_output(output):
    _, top_i = output.data.topk(1)
    return top_i[0]


# returns a python float
def to_scalar(var):
    return var.view(-1).data.tolist()[0]


# return the argmax as a python int
def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def adjust_learning_rate(optimizer, t=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= t


# label转独热编码
def one_hot(y, label_num=2):
    label = torch.LongTensor(np.zeros(label_num)).to(y.device)
    for i in range(label_num):
        if i == float(y[0]):
            label[i] = 1
    return label


def build():
    # SETUP_PATH = app_path()
    PyInstaller.__main__.run([
        '--name=%s' % "main",  # 生成的exe文件名
        ['--onedir', '--onefile'][0],  # 单个目录 or 单个文件
        '--noconfirm',  # Replace output directory without asking for confimation
        ['--windowed', '--console'][1],
        # '--add-binary=./python3.dll;.',  # 外部的包引入
        # '--add-binary=%s' % SETUP_PATH + '/config/logging.yaml;config',  # 配置项
        # '--add-data=%s' % SETUP_PATH + '/config/config.ini;config',  # 分号隔开，前面是添加路径，后面是添加到哪个目录
        # '--hidden-import=%s' % 'sqlalchemy.ext.baked',
        # '--hidden-import=%s' % 'frozen_dir',  # 手动添加包，用于处理 module not found
        'main.py',   # 入口文件
    ])


if __name__ == '__main__':
    import PyInstaller.__main__
    build()