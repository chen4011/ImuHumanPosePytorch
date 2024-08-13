# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np


class HumanBody(object):

    def __init__(self):
        self.skeleton = self.get_skeleton()
        self.skeleton_sorted_by_level = self.sort_skeleton_by_level(
            self.skeleton)
        self.imu_edges = self.get_imu_edges()
        self.imu_edges_reverse = self.get_imu_edges_reverse()

    def get_skeleton(self):
        joint_names = [
            'root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank', 'belly',
            'neck', 'nose', 'lsho', 'lelb', 'lwri', 'rsho', 'relb',  # use nose here instead of head
            'rwri'
        ]
        children = [[1, 4, 7], [2], [3], [], [5], [6], [], [8], [9, 10, 13],
                    [], [11], [12], [], [14], [15], []]
        imubone = [[-1, -1, -1], [3], [4], [], [5], [6], [], [-1], [-1, -1, -1],
                    [], [11], [12], [], [13], [14], []]

        skeleton = []
        for i in range(len(joint_names)):
            skeleton.append({
                'idx': i,
                'name': joint_names[i],
                'children': children[i],
                'imubone': imubone[i]
            })
        return skeleton

    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)

        queue = [skeleton[0]]   # 將骨架的根關節（即第一個關節'idx: 0'）放入隊列
        # print('=> queue:', queue)
        while queue:
            cur = queue[0]
            # print('=> cur:', cur)
            for child in cur['children']:   # 對於 cur 的每個子節點 child，進行以下操作
                skeleton[child]['parent'] = cur['idx']  # 將 cur 的索引設置為 child 的父節點，ex:{'idx': 1, 'name': 'rhip', 'children': [2], 'imubone': [3], 'parent': 0}
                level[child] = level[cur['idx']] + 1    # 將 child 的層級設置為 cur 的層級加一
                queue.append(skeleton[child])   # 將 child 添加到隊列的末尾
            del queue[0]    # 刪除隊列中的第一個元素
            # print('=> queue:', queue)

        desc_order = np.argsort(level)[::-1]    # 將 level 陣列進行降序排序，並獲取排序後的索引
        # print('=> desc_order:', desc_order)
        sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i] # 將 level[i] 設置為 skeleton[i] 的層級
            sorted_skeleton.append(skeleton[i]) # 將 skeleton[i] 添加到 sorted_skeleton 陣列中
        # print('=> sorted_skeleton:', sorted_skeleton)
        return sorted_skeleton

    def get_imu_edges(self):
        # joint to bone
        # imu_edges: {(joint, child): bone_idx}
        imu_edges = dict()
        for joint in self.skeleton:
            # print('=> joint:', joint)
            for idx_child, child in enumerate(joint['children']):
                # print('=>idx_child:', idx_child, 'child:', child)
                if joint['imubone'][idx_child] >= 0:
                    one_edge_name = (joint['idx'], child)
                    # print('=> one_edge_name:', one_edge_name)
                    bone_idx = joint['imubone'][idx_child]
                    # print('=> bone_idx:', bone_idx)
                    imu_edges[one_edge_name] = bone_idx
                    # print('=> imu_edges:', imu_edges)
        return imu_edges

    def get_imu_edges_reverse(self):
        # bone to joint
        imu_edges = self.imu_edges
        imu_edges_reverse = {imu_edges[k]:k for k in imu_edges}
        # print('=> imu_edges_reverse:', imu_edges_reverse)
        return imu_edges_reverse


if __name__ == '__main__':
    hb = HumanBody()
    print(hb.skeleton)
    print(hb.skeleton_sorted_by_level)
