

import argparse

from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


import _init_paths
from models.pose_resnet import PoseResNet, BasicBlock, Bottleneck
from core.config import config
from core.config import update_config
from core.config import update_dir
from utils.vis import save_debug_heatmaps, vis_mid_heatmaps, save_batch_heatmaps


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
    args = parse_args()
    reset_config(config, args)

    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

    num_layers = config.POSE_RESNET.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, config,)
    model.eval()

    # 指定要丟哪些圖片進模型
    image_files = ['/mnt/d/exp_ImuHumanPosePytorch_copy/000100.jpg',
                   '/mnt/d/exp_ImuHumanPosePytorch_copy/000422.jpg',
                   '/mnt/d/exp_ImuHumanPosePytorch_copy/11.jpg',
    ]
    images = [Image.open(file) for file in image_files]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_images = []
    for img in images:
        img_tensor = transform(img)
        batch_images.append(img_tensor)

    batch_tensor = torch.stack(batch_images, dim=0)

    print(batch_tensor.shape)  # (batch_size, 3, 224, 224)
    hms, feature_before_final = model(batch_tensor)
    print(hms.shape) # torch.Size([1, 20, 56, 56])

    # hms = hms.detach().cpu().numpy()  # 將輸出轉換為numpy陣列

    # for i in range(hms.shape[0]):
    #     for j in range(hms.shape[1]):
    #         heatmap = hms[i, j, :, :]  # 取出第i個input的第j個關節的熱圖
    #         plt.imshow(heatmap, cmap='gray')  # use 'gray' colormap for grayscale images
    #         plt.savefig(f'/ssd_sn570/rayhuang/imu-human-pose-pytorch/run/pose2d/{i:06d}_output_{j:02d}.jpg')

    save_batch_heatmaps(batch_image=batch_tensor, batch_heatmaps=hms, file_name='/ssd_sn570/rayhuang/imu-human-pose-pytorch/run/pose2d/hms.jpg')

if __name__ == '__main__':
    main()