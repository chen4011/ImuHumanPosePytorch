# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models
from dataset.totalcapture_collate import totalcapture_collate


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument(
        '--state',
        help='the state of model which is used to test (best or final)',
        default='best',
        type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--model-file', help='model state file', type=str)
    parser.add_argument(
        '--flip-test', help='use flip test', action='store_true')
    parser.add_argument(
        '--post-process', help='use post process', action='store_true')
    parser.add_argument(
        '--shift-heatmap', help='shift heatmap', action='store_true')

    # philly
    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')

    args = parser.parse_args()

    update_dir(args.modelDir, args.logDir, args.dataDir)

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.state:
        config.TEST.STATE = args.state


def main():
    # 解析從 terminal 輸入的參數
    args = parse_args()
    reset_config(config, args)

    # 創建 logger，用於記錄 valid 過程中的訊息
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    # 決定 GPU 是否使用 cuDNN 加速，以及是否使用 cuDNN 的自動調整功能，以提高性能，每次的計算結果可能因當下情況及電腦配置改變
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # 建立機器學習層級架構，使用用於姿態估計的預訓練 ResNet 模型，並設定為不訓練，backbone_model = pose_resnet
    backbone_model = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
        config, is_train=False)
    
    # 用於從多個相機視圖進行姿勢估計的更大系統的一部分，是nn.Module的子類別，得骨架、heatmap參數設定
    model = models.multiview_pose_net.get_multiview_pose_net(backbone_model, config)

    # 載入已經訓練好的模型--res50_256_final.pth.tar 到 model
    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        try:
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE), strict=True)
        except Exception as e:
            logger.info(e)
            logger.info('No worry, still works')
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE), strict=False)
    else:
        model_path = 'final_state_ep{}.pth.tar'.format(config.TEST.STATE)
        model_state_file = os.path.join(final_output_dir, model_path)
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file), strict=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()    # 將模型包裝在 DataParallel 包裝器中，​​並將其移至 GPU

    # 定義損失函數，實際計算損失的地方
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda() #true

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 資料集的紅色、綠色和藍色 (RGB) 通道的平均值和標準差
    
    # 處理和管理與多相機捕捉系統相關的數據
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([transforms.ToTensor(), normalize,]))
        # transforms.Compose([transforms.ToTensor(), normalize,]) 將多個圖像轉換為 Pytporch 張量並歸一化組合在一起
    # valid_dataset = dataset.totalcaptur(config, config.DATASET.TEST_SUBSET = validation, False,
    #                                       transforms.Compose([transforms.ToTensor(), normalize,]))

    # DataLoader 將從 valid_dataset 載入資料
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,  # DataLoader 將從中載入資料的資料集
        batch_size=config.TEST.BATCH_SIZE * len(gpus),  # 每批的樣本數
        shuffle=False,  # 每次迭代時是否打亂數據，對於驗證和測試，通常不打亂資料。
        num_workers=config.WORKERS, # 用於數據加載的子進程數。0表示數據將在主進程中加載。
        collate_fn=totalcapture_collate,    # 將多個樣本組合成一個批次的函數
        pin_memory=True)    # 如果為True，數據將被複製到 CUDA 內存中，然後再從中進行加載。默認值：False

    # evaluate on validation set
    print('=> start validation')
    perf_indicator = validate(config, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)
        # config: 
        # valid_loader: 載入器，從資料集中批次載入資料
        # valid_dataset: 資料集，處理過的 .pkl 檔案
        # model: 要評估的訓練模型
        # criterion: 用於計算模型預測誤差的損失函數
        # final_output_dir: 保存驗證結果的目錄
        # tb_log_dir: 寫入日誌或結果
    logger.info('perf indicator {}'.format(perf_indicator))


if __name__ == '__main__':
    main()
