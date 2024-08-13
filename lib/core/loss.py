# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class JointsMSELoss(nn.Module):
    # 計算某種姿態估計模型中預測關節位置和實際關節位置之間的均方誤差（MSE）損失
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)  # 實際計算損失的地方。 它需要三個參數：輸出（預測的關節位置）、目標（實際的關節位置）和 target_weight（每個關節的權重）。 此方法首先將輸出和目標重塑為具有相同的尺寸，然後將它們分成每個關節的單獨熱圖。 然後計算每個關節的 MSE 損失，如果 self.use_target_weight 為 True，則按 target_weight 加權損失；如果 self.use_target_weight 為 False，則不對損失進行加權。 總損失就是各個聯合損失的總和。
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss
