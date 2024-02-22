# Fusing Wearable IMUs with Multi-View Images for Human Pose Estimation: A Geometric Approach

[Paper](https://www.chunyuwang.org/img/sensor_pose.pdf),

## Installation
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}
2. Install dependencies.
3. Download pytorch imagenet pretrained models. Please download them under ${POSE_ROOT}/models, and make them look like this:

   ```
   ${POSE_ROOT}/models
   └── pytorch
       └── imagenet
           ├── resnet152-b121ed2d.pth
           ├── resnet50-19c8e357.pth
           └── mobilenet_v2.pth.tar
   ```
   They can be downloaded from the following link: [Pretrained Model Download](https://1drv.ms/f/s!AjX41AtnTHeThyJfayggVZSd0M6P)
   


## Data preparation
For **TotalCapture** dataset, please download from [official site](https://cvssp.org/projects/totalcapture/TotalCapture/) and follow [zhezh/TotalCapture-Toolbox](https://github.com/zhezh/TotalCapture-Toolbox) to process data.
>  We have no permission to redistribute this dataset. Please do not ask us for a copy.

For **precalculated pictorial model pairwise** term, please download from [HERE](https://dllabml-my.sharepoint.com/:f:/g/personal/research_dllabml_onmicrosoft_com/EtF51b86YvdEvcwErjkluGsBVbQXeXfMTUNEfc04BsNNDA?e=pMK1s9), and save in `data/pict`.

To reproduce our results in the paper, please download the trained models from [HERE](https://dllabml-my.sharepoint.com/:f:/g/personal/research_dllabml_onmicrosoft_com/EjpV84hHu0RGmiLl_3BjpWMBK1S15OzygM0pNxnf7dLevw?e=bYTlCV).

## Testing
### Testing *ORN*
1. 需要額外安裝 charset_normalizer 的 site package，`pip install charset_normalizer` 
2. 在終端機執行以下命令，記得確認每一個要用到的檔案如以下位置放置
    ```
    python run/pose2d/valid.py \
    --cfg experiments-local/totalcapture/res50_256_orn.yaml \
    --model-file <path-to-your-download>/res50_256_final.pth.tar \
    --gpus 0 --workers 1 \
    --dataDir . --logDir log --modelDir output 
    ```
3. 得到結果在 output\totalcapture\multiview_pose_resnet_50\res50_256_orn

### Testing *ORPSM*
1. 需要額外安裝 sklearn 的 site package， `pip install scikit-learn`
2. 在終端機執行以下命令，記得確認每一個要用到的檔案如以下位置放置
    
    ```
    python run/pose3d/estimate.py \
    --cfg experiments-local/totalcapture/res50_256_orn.yaml \
    --withIMU 1 \
    --dataDir . --logDir log --modelDir output
    ```
    
3. 得到結果在 `output\totalcapture\multiview_pose_resnet_50\res50_256_orn`

### Testing Baseline (SN + PSM)
1. 在終端機執行以下命令，記得確認每一個要用到的檔案如以下位置放置
    
    ```
    python run/pose2d/valid.py \
    --cfg experiments-local/totalcapture/res50_256_nofusion.yaml \
    --model-file /mnt/d/exp_ImuHumanPosePytorch/res50_256_final.pth.tar \
    --gpus 0 --workers 1 \
    --dataDir . --logDir log --modelDir output
    ```
    
2. 得到結果在 `output\totalcapture\multiview_pose_resnet_50\res50_256_nofusion`

Then,
1. 在終端機執行以下命令，記得確認每一個要用到的檔案如以下位置放置
    
    ```
    python run/pose3d/estimate.py \
    --cfg experiments-local/totalcapture/res50_256_nofusion.yaml \
    --withIMU 0 \
    --dataDir . --logDir log --modelDir output
    ```
    
2. 得到結果在 `output\totalcapture\multiview_pose_resnet_50\res50_256_nofusion`

## Training
Since our ORN and ORPSM has no learnable parameters, it can be conveniently appended to any 2D pose estimator. Thus training the SN backbone is sufficient.
```
python run/pose2d/train.py \
--cfg experiments-local/totalcapture/res50_256_nofusion.yaml \
--gpus 0 --workers 1 \
--dataDir . --logDir log --modelDir output
```

## Citation
```
@inproceedings{zhe2020fusingimu,
  title={Fusing Wearable IMUs with Multi-View Images for Human Pose Estimation: A Geometric Approach},
  author={Zhang, Zhe and Wang, Chunyu and Qin, Wenhu and Zeng, Wenjun},
  booktitle = {CVPR},
  year={2020}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

