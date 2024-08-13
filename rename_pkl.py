import torch
import os

# 提供文件的完整路徑
file_path = '/mnt/d/exp_ImuHumanPosePytorch_copy/occlusion_person_8view.pth.tar'

# 確認文件存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# 加载模型
checkpoint = torch.load(file_path)

# 打印checkpoint的所有键
print("Checkpoint keys:", checkpoint.keys())

# 创建一个新的字典来保存修改后的权重
new_state_dict = {}

# 遍历所有层并重新命名
for key in checkpoint.keys():
    new_key = 'resnet.' + key  # 在每个层名前添加前缀 'resnet.'
    new_state_dict[new_key] = checkpoint[key]

# 将修改后的权重保存到新的checkpoint
torch.save(new_state_dict, '/mnt/d/exp_ImuHumanPosePytorch_copy/occlusion_person_8view_renamed.pth.tar')

print("已成功修改層名稱並保存新的模型檔案。")

# 確認修改後名稱
file_path = '/mnt/d/exp_ImuHumanPosePytorch_copy/occlusion_person_8view_renamed.pth.tar'
checkpoint = torch.load(file_path)
print("Checkpoint keys:", checkpoint.keys())